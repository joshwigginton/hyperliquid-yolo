import pandas as pd
import requests
import re
import ccxt
from hyperliquid.info import Info
from hyperliquid.utils import constants
import example_utils
import base64
import json
import logging
import time

# Version 1.0 Main function deployed to GCP Cloud function. Added "No Yolo Traades" when filtered DF is empty
# Version 1.1 Added 500 error handler, wait 3 seconds and retry. 
# Version 1.2 Fixed bug where it would duplicating orders due to duplicate code. 
# Version 1.3 Added a function check_and_modify_orders, timer to cancel unfilled open orders and then create market orders.

def setup_logging():
    """
    Configure the logging system to write logs to a file named with the current date.
    
    Logs include the timestamp, log level, and log message, saved to a file that helps in auditing and debugging.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('log_live.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PortfolioManager:
    """
    PortfolioManager class manages the portfolio and executes trades based on YOLO weights and HyperLiquid positions.
    """

    def __init__(self, cash, buffer, MM, TM, ML, rw_api_key, wallet_address,sandbox=False):
        """
        Initialize PortfolioManager with portfolio parameters and API key.

        Parameters:
        - cash: Initial cash amount.
        - buffer: Buffer for trade execution.
        - MM: Momentum factor.
        - TM: Trend factor.
        - ML: Maximum leverage.
        - api_key: API key for accessing YOLO weights.
        """
        self.cash = cash
        self.buffer = buffer
        self.MM = MM
        self.TM = TM
        self.ML = ML
        self.rw_api_key = rw_api_key
        self.wallet_address = wallet_address
        self.ccxt_exchange = ccxt.hyperliquid({'enableRateLimit': True})
        self.ccxt_exchange.load_markets()
        
              
        # This requires hyperliquid example_utils from the python SDK here.Config MAINNET or TESTNET 
        # Need config.json file with your hyperliquid wallet and secret key.
        # Set the base URL based on the sandbox parameter
        base_url = constants.TESTNET_API_URL if sandbox else constants.MAINNET_API_URL               
        self.address, self.info, self.hyperliquid_exchange = example_utils.setup(base_url=base_url, skip_ws=True)

    def get_yolo_weights(self):
        """
        Fetch YOLO weights from the API.

        Returns:
        - DataFrame: DataFrame containing YOLO weights.
        """
        try:
            url = f"https://api.robotwealth.com/v1/yolo/weights?api_key={self.rw_api_key}"
            json_data = requests.get(url).json()
            yoloW = pd.json_normalize(json_data['data'])
            yoloW['ticker'] = yoloW['ticker'].apply(lambda x: re.sub(r'/USD', '', x))
            logging.info("YOLO Weights fetched")
            print("YOLO Weights fetched")
            return yoloW
        except Exception as e:
            logging.error(f"Error fetching YOLO weights: {e}")
            return None

    def get_hyperliquid_positions(self):
        """
        Fetch HyperLiquid positions.

        Returns:
        - DataFrame: DataFrame containing HyperLiquid positions.
        """
        try:
            user_state = self.info.user_state(self.wallet_address)
            asset_positions = user_state.get("assetPositions", [])
            positions_info = [{"ticker": position["position"]["coin"] + "USDT", "current_position": position["position"]["szi"]} for position in asset_positions]
            hyperliquid_positions = pd.DataFrame(positions_info)
            logging.info("HyperLiquid positions fetched")
            print("HyperLiquid positions fetched")
            return hyperliquid_positions
        except Exception as e:
            logging.error(f"Error fetching HyperLiquid positions: {e}")
            return None

    def merge_and_process_data(self):
        """
        Merge YOLO weights and HyperLiquid positions, and process the data to calculate trades.

        Returns:
        - DataFrame: DataFrame containing processed data.
        """
        try:
            # Instantiate the DataFrame with required columns
            yoloT = pd.DataFrame(columns=['ticker', 'date', 'arrival_price', 'combo_weight', 'momentum_megafactor', 'trend_megafactor', 'current_position'])
        
            # Merge YOLO weights and HyperLiquid positions
            yoloW = self.get_yolo_weights()
            hyperliquid_positions = self.get_hyperliquid_positions()
            
            # Merge the data into the DataFrame
            if not yoloW.empty:
                if not hyperliquid_positions.empty:
                    yoloT = self.merge_yolo_hyperliquid(yoloW, hyperliquid_positions)
                else:
                    logging.info("HyperLiquid positions are empty. Using YOLO weights only.")
                    print("HyperLiquid positions are empty. Using YOLO weights only.")
                    yoloT = yoloW
                
                # Print warning for tickers in hyperliquid_positions not present in yoloW
                if not hyperliquid_positions.empty:
                    hyperliquid_tickers = set(hyperliquid_positions['ticker'])
                    yoloW_tickers = set(yoloW['ticker'])
                    for ticker in hyperliquid_tickers - yoloW_tickers:
                        logging.warning(f"{ticker} is in HyperLiquid positions but not in Yolo Weights")
                        print(f"WARNING: {ticker} is in HyperLiquid positions but not in Yolo Weights")
            
                # Add 'current_position' column with 0 values if not present
                if 'current_position' not in yoloT.columns:
                    yoloT['current_position'] = 0
            else:
                logging.error("YOLO weights are empty. Unable to proceed.")
            
            if yoloT is not None:
                # Fill NaN values in 'current_position' column with 0
                yoloT['current_position'] = yoloT['current_position'].fillna(0)

                yoloT = self.calculate_unconstrTW(yoloT)
                yoloT = self.calculate_constrTW(yoloT)
                yoloT = self.handle_missing_price_information(yoloT)

                logging.info("Data merged and processed.")
                print("Data merged and processed.")
                return yoloT
            else:
                logging.error("Error merging and processing data.")
                return None
            
        except Exception as e:
            logging.error(f"Error merging and processing data: {e}")
            return None

    def merge_yolo_hyperliquid(self, yoloW, hyperliquid_positions):
        """
        Merge YOLO weights and HyperLiquid positions.

        Parameters:
        - yoloW: DataFrame containing YOLO weights.
        - hyperliquid_positions: DataFrame containing HyperLiquid positions.

        Returns:
        - DataFrame: Merged DataFrame.
        """
        try:
            yoloT = pd.merge(yoloW, hyperliquid_positions, on='ticker', how='left')
            return yoloT
        except Exception as e:
            logging.error(f"Error merging dataframes: {e}")


    def calculate_unconstrTW(self, yoloT):
        """
        Calculate unconstrained target weights.

        Parameters:
        - yoloT: DataFrame containing merged data.

        Returns:
        - DataFrame: DataFrame with 'unconstrTW' column added.
        """
        try:
            yoloT['unconstrTW'] = ((yoloT['momentum_megafactor'].astype(float) * self.MM) + 
                                   (yoloT['trend_megafactor'].astype(float) * self.TM)) / 2
            return yoloT
        except Exception as e:
            logging.error(f"Error calculating unconstrained target weights: {e}")
            return None

    def calculate_constrTW(self, yoloT):
        """
        Calculate constrained target weights.

        Parameters:
        - yoloT: DataFrame containing merged data.

        Returns:
        - DataFrame: DataFrame with 'constrTW' column added.
        """
        try:
            unconstrTL = abs(yoloT['unconstrTW']).sum()
            yoloT['constrTW'] = yoloT.apply(lambda x: x['unconstrTW'] if unconstrTL < self.ML else (x['unconstrTW'] * self.ML / unconstrTL), axis=1)
            return yoloT
        except Exception as e:
            logging.error(f"Error calculating constrained target weights: {e}")
            return None

    def handle_missing_price_information(self, yoloT):
        """
        Handle missing price information.

        Parameters:
        - yoloT: DataFrame containing merged data.

        Returns:
        - DataFrame: DataFrame with additional calculated columns.
        """
        try:
            if 'arrival_price' in yoloT.columns:
                yoloT['currentPosValue'] = yoloT['arrival_price'].astype(float) * yoloT['current_position'].astype(float)
                yoloT['currentW'] = yoloT['current_position'].astype(float) * yoloT['arrival_price'].astype(float) / self.cash

                def calculate_trades(row):
                    if row['currentW'] < row['constrTW'] - self.buffer:
                        return (row['constrTW'] - self.buffer - row['currentW']) * self.cash / row['arrival_price']
                    elif row['currentW'] > row['constrTW'] + self.buffer:
                        return (row['constrTW'] + self.buffer - row['currentW']) * self.cash / row['arrival_price']
                    else:
                        return 0

                yoloT['trades'] = yoloT.apply(calculate_trades, axis=1)
                yoloT['tradeValue'] = yoloT['trades'] * yoloT['arrival_price'].astype(float)
                yoloT['PtPos'] = yoloT['current_position'].astype(float) + yoloT['trades']
                yoloT['PtPosValue'] = yoloT['PtPos'] * yoloT['arrival_price'].astype(float)
                yoloT['PtW'] = yoloT['PtPosValue'] / self.cash
                yoloT['DiffToCW'] = yoloT['constrTW'] - yoloT['currentW']
            else:
                logging.error("'arrival_price' information is missing in the DataFrame.")
            return yoloT
        except Exception as e:
            logging.error(f"Error handling missing price information: {e}")
            return None

    def execute_trades(self, df):
        """
        Execute trades based on calculated trades DataFrame.

        Parameters:
        - df: DataFrame containing trades information.

        Returns:
        - None
        """
        try:
            for index, row in df.iterrows():
                target_error = "Post only order would have immediately matched"
                ticker = row['ticker'].replace('USDT', '')
                order_amount = abs(row['trades'])
                order_amount = float(self.ccxt_exchange.amount_to_precision(ticker + "/USDC:USDC", abs(row['trades'])))

                while True:
                    order_params = {
                        'coin': ticker,
                        'is_buy': (row['trades'] > 0),
                        'sz': order_amount,
                        'limit_px': self.get_mid_price(ticker),
                        'order_type': {'limit': {'tif': 'Alo'}}
                    }

                    logging.info(order_params)
                    print(order_params)

                    try:
                        order_result = self.hyperliquid_exchange.order(**order_params)
                        logging.info(order_result)
                        print(order_result)
                    except Exception as e:
                        if '500' in str(e):  # Check if the error message contains '500'
                            logging.error(f"Encountered 500 server error. Retrying after 3 seconds...")
                            print("Encountered 500 server error. Retrying after 3 seconds...")
                            time.sleep(3)  # Wait for 3 seconds before retrying
                            continue
                        else:
                            raise e

                    if 'error' in order_result['response']['data']['statuses'][0] and \
                           target_error in order_result['response']['data']['statuses'][0]['error']:
                        logging.info("Retrying order due to Post Only matched error...")
                        print("Retrying order due to Post Only matched error...")
                        continue    # Retry the order if the error message matches the target error

                    if 'resting' in order_result['response']['data']['statuses'][0]:
                        status = order_result['response']['data']['statuses'][0]
                        order_status = self.info.query_order_by_oid(self.address, status['resting']['oid'])
                        logging.info("Order status by oid:", order_status)
                        print("Order status by oid:", order_status)
                    break

        except Exception as e:
            logging.error(f"Error executing trades: {e}")
        

    def get_mid_price(self, ticker):
        """
        Get the mid price for a given ticker.

        Parameters:
        - ticker: Ticker symbol.

        Returns:
        - float: Mid price.
        """
        try:
            mids = self.info.all_mids()
            mid_price = float(mids[ticker])
            return float(f"{mid_price:.5g}")
        except Exception as e:
            logging.error(f"Error getting mid price: {e}")
            return None
    
    def check_and_modify_orders(self, sleep_time=300):
        """
        Checks for open orders, waits sleep timer, and if still open - cancels and creates market orders.

        Parameters:
        - sleep_time: Sleep Timer in seconds, defaults to 300.

        """
        
        # Wait for 60 seconds initially
        time.sleep(60)

        # Check for open orders
        open_orders = self.info.open_orders(self.wallet_address)

        if not open_orders:
            # No open orders found after 60 seconds, all orders were filled quickly
            print("No open orders found. All orders were filled quickly.")
            logging.info("No open orders found. All orders were filled quickly.")
            return

        # Wait for remaining sleep time, default 300 seconds
        print(f"Waiting for {sleep_time} seconds for orders to fill.")
        logging.info(f"Waiting for {sleep_time} seconds for orders to fill.")
        time.sleep(sleep_time)  

        # Check for open orders again
        open_orders = self.info.open_orders(self.wallet_address)

        if open_orders:
            # If there are still open orders after sleep time, proceed to cancel and create market orders
            for open_order in open_orders:
                oid = open_order['oid']
                coin = open_order['coin']
                is_buy = True if open_order['side'] == 'B' else False
                sz = float(open_order['sz'])

                # Cancel the order
                print(f"cancelling order {open_order}")
                logging.info(f"cancelling order {open_order}")
                cancel_result = self.hyperliquid_exchange.cancel(coin, oid)
                if cancel_result['status'] == 'ok':
                    logging.info(f"Order {oid} for {coin} cancelled successfully.")
                    print(f"Order {oid} for {coin} cancelled successfully.")

                    # Check if the cancellation status is 'success'
                    if 'success' in cancel_result['response']['data']['statuses']:
                        # Create a market order
                        order_result = self.hyperliquid_exchange.market_open(coin, is_buy, sz, None, 0.01)
                        if order_result["status"] == "ok":
                            for status in order_result["response"]["data"]["statuses"]:
                                try:
                                    filled = status["filled"]
                                    print(f'Market Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                                    logging.info(f'Market Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                                except KeyError:
                                    print(f'Error: {status["error"]}')
                                    logging.info(f'Error: {status["error"]}')
                    else:
                        statuses = cancel_result['response']['data']['statuses']
                        status_info = ', '.join([f"{status.get('status', 'Unknown status')} ({status.get('message', 'No message')})" for status in statuses if isinstance(status, dict)])
                        print(f"Failed to cancel order {oid} for {coin}. Status: {status_info}")
                        logging.info(f"Failed to cancel order {oid} for {coin}. Status: {status_info}")
                else:
                    print(f"Failed to cancel order {oid} for {coin}. Error: {cancel_result.get('error', 'Unknown error')}")
                    logging.info(f"Failed to cancel order {oid} for {coin}. Error: {cancel_result.get('error', 'Unknown error')}")
        else:
            print(f"No open orders found after waiting {sleep_time} seconds. All orders were filled.")
            logging.info(f"No open orders found after waiting {sleep_time} seconds. All orders were filled.")


def hello_pubsub(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    
    # Load configuration from config.json
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)
        cash = config_data.get('cash')
        buffer = config_data.get('buffer')
        MM = config_data.get('MM')
        TM = config_data.get('TM')
        ML = config_data.get('ML')
        rw_api_key = config_data.get('rw_api_key')
        wallet_address = config_data.get('wallet_address')

    """
    The main function to execute the Yolo portfolio rebalance.
    
    It creates an instance of the PortfolioManager class, process RW yolo weights and target allocations,
    calculates necessary orders, and executes them accordingly.
    """
        
    portfolio_manager = PortfolioManager(cash, buffer, MM, TM, ML, rw_api_key, wallet_address)
    processed_data = portfolio_manager.merge_and_process_data()
    if processed_data is not None:
        # This filtered DF is the trades to get within 2% and also have a tradeValue of $100 or more.
        filtered_df = processed_data[(processed_data['trades'] != 0) & (abs(processed_data['tradeValue']) > 100)]
        yolo_daily_trades = filtered_df[['ticker', 'trades']]
        if yolo_daily_trades.empty:
            logging.info("No yolo trades to rebalance.")
            print("No yolo trades to rebalance.")
        else:
            logging.info(yolo_daily_trades)
            print(yolo_daily_trades)
            portfolio_manager.execute_trades(yolo_daily_trades)
            print("Checking if all orders are filled")
            logging.info("Checking if all orders are filled")
            portfolio_manager.check_and_modify_orders()
        logging.info("Yolo Rebalancing complete.")
        print("Yolo Rebalancing complete.")
    print('Pub/sub message:', pubsub_message)