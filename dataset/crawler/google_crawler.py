import os
import argparse

import requests
from bs4 import BeautifulSoup

google_image = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"

user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
}


def download_images(search_queries, num_images, save_path):
    count = 0
    for search_query in search_queries:
        search_url = google_image + 'q=' + search_query
        print(search_url)
        response = requests.get(search_url, headers=user_agent)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        results = soup.findAll('img', {'class': 'rg_i Q4LuWd'})

        links = []
        for result in results:
            try:
                link = result['data-src']
                links.append(link)
            except KeyError:
                continue

        print(f"Downloading {len(links)} images...")
        for i, link in enumerate(links):
            response = requests.get(link)
            image_name = os.path.join(save_path, search_query.split('+')[0] + "_" + str(count) + '.jpg')
            count += 1
            with open(image_name, 'wb') as fh:
                fh.write(response.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    # parser.add_argument('--search_query', type=str, nargs='+', required=True, help='Query to search')
    parser.add_argument('--num_images', type=int, default=500, required=False, help='Number of images to download')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the images')
    args = parser.parse_args()
    # args.search_query = '+'.join(args.search_query)
    print(args)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    axe_queries = ['axe', 'axe+real+life', 'fireaxe', 'axe+fight', 'axe+real+life+fight']
    machete_queries = ['machete', 'machete+real+life', 'machete+danger', 'machete+fight', 'machete+real+life+fight']
    shotgun_queries = ['shotgun', 'shotgun+real+life', 'shotgun+shooting', 'shotgun+fight',
                       'shotgun+real+life+shooting']
    submachine_gun_queries = ['submachine+gun', 'submachine+gun+real+life', 'submachine+gun+shooting',
                              'submachine+gun+fight', 'submachine+gun+real+life+shooting']

    download_images(axe_queries, args.num_images, args.save_path)
    download_images(machete_queries, args.num_images, args.save_path)
    download_images(shotgun_queries, args.num_images, args.save_path)
    download_images(submachine_gun_queries, args.num_images, args.save_path)
