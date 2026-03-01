from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://books.toscrape.com/"
response = requests.get(url)
print(response.status_code)
soup = BeautifulSoup(response.text, "html.parser")
books = soup.find_all("article", class_="product_pod")
for book in books:
    title = book.h3.a["title"]
    price = book.find("p", class_="price_color").text
    rating = book.find("p", class_="star-rating")["class"][1]
    availability = book.find("p", class_="instock availability").text.strip()
    print(f"{title} - {price} - {rating} - {availability}")

Books_data = [
{
"Title": book.h3.a["title"],
"Price": book.find("p", class_="price_color").text,
"Rating": book.find("p", class_="star-rating")["class"][1],
"Availability": book.find("p", class_="instock availability").text.strip()
}
for book in books
]
df = pd.DataFrame(Books_data)
print(df.head())