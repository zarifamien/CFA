import os
import time
import requests
import pandas as pd
import argparse
from datetime import datetime, timezone, timedelta
from colorama import Fore, init

init(autoreset=True)

def clear_screen():
  """
  Clear the console screen.
  """
  os.system("cls" if os.name == "nt" else "clear")

def fetch_bitstamp_data(currency_pair, start_timestamp, end_timestamp, step=60, limit=1000, verbose=False):
  """
  Fetch OHLC (Open, High, Low, Close) data from Bitstamp for a given currency pair.

  Args:
    currency_pair (str): The currency pair to fetch data for (e.g., 'btcusd').
    start_timestamp (int): The start timestamp in Unix time.
    end_timestamp (int): The end timestamp in Unix time.
    step (int, optional): The interval in seconds between data points. Default is 60 seconds.
    limit (int, optional): The maximum number of data points to fetch. Default is 1000.
    verbose (bool, optional): Whether to print additional information. Default is False.

  Returns:
    list: A list of OHLC data points, each represented as a dictionary.
  """
  url = f"https://www.bitstamp.net/api/v2/ohlc/{currency_pair}/"
  params = {
    "step": step,
    "start": start_timestamp,
    "end": end_timestamp,
    "limit": limit,
  }
  try:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    if verbose:
      print(f"{Fore.GREEN}       > Fetched Data From {datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")

    return response.json().get("data", {}).get("ohlc", [])
  except requests.exceptions.RequestException as e:
    print(f"{Fore.RED}[ERROR] Fetching Data: {e}")
    return []

def fetch_and_save_crypto_data(currency_pair, start_date, output_filename, step, verbose=False):
    """
    Fetch and save OHLC data for a specific cryptocurrency to a CSV file.

    Args:
      currency_pair (str): The currency pair to fetch data for (e.g., 'btcusd').
      start_date (datetime.date): The start date for fetching data.
      output_filename (str): The filename to save the fetched data to.
      step (int): The interval in seconds between data points.
      verbose (bool, optional): Whether to print additional information. Default is False.

    Returns:
    bool: True if data was fetched and saved successfully,
          False if no data was found for the given currency pair.
    """
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(datetime.now(timezone.utc).timetuple()))
    limit = 1000
    all_data = []

    current_start = start_timestamp
    while current_start < end_timestamp:
      current_end = min(current_start + (step * limit), end_timestamp)
      new_data = fetch_bitstamp_data(currency_pair, current_start, current_end, step, limit, verbose)
      if new_data: all_data.extend(new_data)
      current_start = current_end

    if all_data:
      df_new = pd.DataFrame(all_data)
      df_new["timestamp"] = pd.to_numeric(df_new["timestamp"], errors="coerce")
      df_new.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
      df_new.to_csv(output_filename, index=False)
      return True
    else:
      return False
    
def main(step, days=100, verbose=False):
  """
  Main execution block. Fetches and saves OHLC data for a list of cryptocurrencies.

  Args:
    days (int, optional): The number of days to fetch data for. Default is 100.
    verbose (bool, optional): Whether to print additional information. Default is False.
  """
  
  clear_screen()

  # the list of cryptocurrencies and their pairs on Bitstamp
  cryptocurrencies = {
    "Bitcoin": "btcusd",
    "Ethereum": "ethusd",
    "XRP": "xrpusd",
    "Solana": "solusd",
    "DogeCoin": "dogeusd",
    "Cardano": "adausd",
    "Tether": "usdtusd",
    "USDC": "usdcusd",
    "DAI": "daiusd",
    "Algorand": "algousd",
    "Litecoin": "ltcusd",
  }

  if not os.path.exists("data"): 
    os.makedirs("data")

  start_date = datetime.now(timezone.utc) - timedelta(days=days)
  print(f'{Fore.CYAN}[INFO] Fetching Data From {start_date.strftime("%Y-%m-%d")} to {datetime.now(timezone.utc).strftime("%Y-%m-%d")} (last {days} days)')

  # to fetch and save data for each cryptocurrency
  for crypto_name, currency_pair in cryptocurrencies.items():
    start_time = time.time()
    print(f"{Fore.GREEN}[INFO] Fetching Data For '{crypto_name}' ({currency_pair})...")
    output_filename = os.path.join("data", f"{currency_pair}.csv")
    status = fetch_and_save_crypto_data(currency_pair, start_date, output_filename, step, verbose)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if status:
      print(f"{Fore.GREEN}       > Data Saved To '{output_filename}' (Time taken: {elapsed_time:.2f} seconds)")
    else:
      print(f"{Fore.YELLOW}[INFO] No Data Found for '{currency_pair}'")

  print(f"{Fore.CYAN}[INFO] Data Fetching Completed!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="CryptoTrend (Phase-I: Data Acquisition)")
  parser.add_argument("-d", "--days", type=int, default=100, help="Number of days to fetch data for (default: 100)")
  parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
  parser.add_argument("-s", "--step", type=int, default=60, help="Interval in seconds between data points (default: 60)")

  args = parser.parse_args()
  main(
    args.step,
    args.days, 
    args.verbose
  )