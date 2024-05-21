# YOLO Crypto Trading Bot

## Overview

This project implements a Python-based trading application that integrates with the Robot Wealth YOLO Crypto strategy. The strategy is based on Cross-Sectional and Time-Series momentum factors and is designed to execute trades and rebalance the portfolio weights through the Hyperliquid DEX. 

- For more info on Robot Wealth, check out https://robotwealth.com/
- For more info on Hyperliquid Exchange, check out https://app.hyperliquid.xyz/

## Features

- Fetches YOLO weights using the Robot Wealth API.
- Retrieves positions from the Hyperliquid DEX.
- Calculates and executes trades based on the YOLO strategy.
- Handles trade execution with market and limit orders.
- Includes error handling and logging for better auditing and debugging.
- Deployed as a Google Cloud Function triggered by Pub/Sub messages and scheduled via Cloud Scheduler.
- example_utils.py from hyperliquid python SDK. https://github.com/hyperliquid-dex/hyperliquid-python-sdk

## Version History

- Version 1.0: Main function deployed to GCP Cloud Function. Added "No Yolo Trades" message when filtered DataFrame is empty.
- Version 1.1: Added 500 error handler with a retry mechanism.
- Version 1.2: Fixed a bug where duplicate orders were created due to duplicate code.
- Version 1.3: Added a function to cancel unfilled open orders after a specific time and create market orders.

## Setup

### Prerequisites

- Python 3.x
- Google Cloud SDK
- Git
- Virtualenv (optional but recommended)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `config.json` file with your configuration parameters. See example_config.json :
    ```json
    {
        "cash": 10000,
        "buffer": 0.02,
        "MM": 0.5,
        "TM": 0.5,
        "ML": 2,
        "rw_api_key": "your_robot_wealth_api_key",
        "wallet_address": "your_hyperliquid_wallet_address"
    }
    ```

## Usage

### Running Locally

To run the application locally, you can invoke the main function directly. Make sure to set up your logging and configuration properly.

### Deploying to Google Cloud

1. **Create a Cloud Function:**

    ```sh
    gcloud functions deploy yolo-crypto-trading-bot \
    --runtime python39 \
    --trigger-topic YOUR_TOPIC_NAME \
    --entry-point hello_pubsub \
    --memory 512MB \
    --timeout 540s
    ```

2. **Set Up Google Cloud Pub/Sub:**

    - Create a Pub/Sub topic:
        ```sh
        gcloud pubsub topics create YOUR_TOPIC_NAME
        ```

    - Create a subscription to the topic (optional):
        ```sh
        gcloud pubsub subscriptions create YOUR_SUBSCRIPTION_NAME --topic YOUR_TOPIC_NAME
        ```

3. **Set Up Cloud Scheduler:**

    - Create a Cloud Scheduler job to trigger the Cloud Function:
        ```sh
        gcloud scheduler jobs create pubsub yolo-crypto-scheduler-job \
        --schedule="0 */1 * * *" \
        --time-zone="UTC" \
        --topic=YOUR_TOPIC_NAME \
        --message-body="{}"
        ```

## Code Description

### Main Components

- **PortfolioManager Class:**
    - Manages the portfolio and executes trades based on YOLO weights and Hyperliquid positions.
    - Methods include fetching YOLO weights and Hyperliquid positions, merging and processing data, calculating target weights, handling price information, and executing trades.

- **hello_pubsub Function:**
    - Triggered by Pub/Sub messages.
    - Loads configuration, creates an instance of `PortfolioManager`, processes data, and executes trades.

### Key Functions

- `setup_logging()`: Configures logging to write logs to a file.
- `get_yolo_weights()`: Fetches YOLO weights from the Robot Wealth API.
- `get_hyperliquid_positions()`: Retrieves current positions from Hyperliquid.
- `merge_and_process_data()`: Merges YOLO weights with Hyperliquid positions and processes the data.
- `execute_trades()`: Executes the trades based on the processed data.
- `check_and_modify_orders()`: Sleep timer to check if orders have been filled, if not will execute market orders, needs some refactoring and future imprmovements.  

## Contributing
Contributions are welcome! If you have suggestions, feature requests, or find a bug, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.