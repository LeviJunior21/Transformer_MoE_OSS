import os
import requests
from tqdm import tqdm

from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import time


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

url = "https://www.gutenberg.org/browse/languages/pt"
driver.get(url)

divs = driver.find_element(By.CLASS_NAME, "pgdbbylanguage")
headers = divs.find_elements(By.TAG_NAME, "h2")
uls = divs.find_elements(By.TAG_NAME, "ul")

livros = []
for h2, ul in zip(headers, uls):
    lis = ul.find_elements(By.TAG_NAME, "li")
    for li in lis:
        if not "English" in li.text:
            a = li.find_element(By.TAG_NAME, "a")
            codigo = a.get_attribute('href').split("/")[-1]
            link = f"https://www.gutenberg.org/files/{codigo}"
            print(f"📄 {link}")
            livros.append(link)

i = 0
BOOKS = {}
for livro in livros:
    try:
        driver.get(livro)
        time.sleep(1)

        anchors = driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            text = a.text.strip()
            href = a.get_attribute("href")
            if text.endswith(".txt") or (href and href.endswith(".txt")):
                BOOKS[str(i)] = href
                print(f"📄Baixando {href}")
                i += 1
                break
    except Exception as e:
        print(f"❌ Erro em {livro}: {e}")
driver.quit()


OUTPUT_DIR = "data/raw"

def download_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_bytes_downloaded = 0

    for filename, url in BOOKS.items():
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Baixando {filename}")
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        chunk_size = len(chunk)
                        downloaded += chunk_size
                        progress_bar.update(chunk_size)
            
            progress_bar.close()
            total_bytes_downloaded += downloaded

            if total_size != 0 and progress_bar.n != total_size:
                print(f"⚠️ AVISO: O tamanho do download de {filename} não corresponde ao esperado.")
            else:
                print(f"✅ {filename} baixado ({downloaded / 1024 / 1024:.2f} MB)")

        except requests.exceptions.RequestException as e:
            print(f"❌ ERRO ao baixar {filename}: {e}")

    print("\n📦 Download de todos os arquivos concluído!")
    total_mb = total_bytes_downloaded / 1024 / 1024
    if total_mb < 1024:
        print(f"💾 Tamanho total baixado: {total_mb:.2f} MB")
    else:
        print(f"💾 Tamanho total baixado: {total_mb / 1024:.2f} GB")


if __name__ == "__main__":
    download_files()