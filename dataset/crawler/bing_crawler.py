import os
import argparse

from bs4 import BeautifulSoup
import json
import urllib.request, urllib.error, urllib.parse


bing_template_search = "http://www.bing.com/images/search?"  # q=" + query + "&FORM=HDRSC2"

user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
}


def get_soup(url, header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header)), 'html.parser')


def download_images(search_queries, num_images, save_path):
    count = 0
    for search_query in search_queries:
        search_url = bing_template_search + 'q=' + search_query + "&FORM=HDRSC2"
        print(search_url)
        soup = get_soup(search_url, user_agent)

        links = []  # contains the link for Large original images, type of  image
        for a in soup.find_all("a", {"class": "iusc"}):
            # mad = json.loads(a["mad"])
            # turl = mad["turl"]
            # m = json.loads(a["m"])
            # murl = m["murl"]
            m = json.loads(a["m"])
            murl, turl = m["murl"], m["turl"]

            image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]
            print(image_name, turl, murl)
            links.append((image_name, turl, murl))

        print("there are total", len(links), "images")

        for i, (image_name, turl, murl) in enumerate(links):
            try:
                # req = urllib2.Request(turl, headers={'User-Agent' : header})
                # raw_img = urllib2.urlopen(req).read()
                # req = urllib.request.Request(turl, headers={'User-Agent' : header})
                raw_img = urllib.request.urlopen(turl).read()

                f = open(os.path.join(save_path, search_query.split('+')[0] + "_" + str(count) + '.jpg'), 'wb')
                f.write(raw_img)
                f.close()
                count += 1
            except Exception as e:
                print("could not load : " + image_name)
                print(e)


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

    axe_queries = ['axe', 'axe+real+life', 'fireaxe', 'axe+fight', 'axe+real+life+fight', 'axe+use']
    machete_queries = ['machete', 'machete+real+life', 'machete+danger', 'machete+fight', 'machete+real+life+fight',
                       'machete+use']
    shotgun_queries = ['shotgun', 'shotgun+real+life', 'shotgun+shooting', 'shotgun+fight',
                       'shotgun+real+life+shooting', 'shotgun+use']
    submachine_gun_queries = ['submachine+gun', 'submachine+gun+real+life', 'submachine+gun+shooting',
                              'submachine+gun+fight', 'submachine+gun+real+life+shooting',
                              'submachine+gun+use']

    download_images(axe_queries, args.num_images, args.save_path)
    download_images(machete_queries, args.num_images, args.save_path)
    download_images(shotgun_queries, args.num_images, args.save_path)
    download_images(submachine_gun_queries, args.num_images, args.save_path)
