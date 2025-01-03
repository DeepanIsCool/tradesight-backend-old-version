import requests
from bs4 import BeautifulSoup
import time

ticker = 'TCS' #TCS, 
exchange = 'NSE' #NSE, BOM

url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"

while True:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    price = float(soup.find('div', {'class': 'YMlKec fxKbKc'}).text.strip()[1:].replace(',',''))
    print(price)
    time.sleep(10)