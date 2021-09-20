import json

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def read_json(input_file):
    files = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        for p in data['input']:
            if 'src' in p.keys():
                if p['src'].split('.')[-1] in img_formats + vid_formats:
                    files.append(p['src'])
    return files


def save_retrieved_ranking(query_image, ranking, bb, save_path):
    """
    Function to save the top 10 visual results of the query
    Person retrieval considers the most similar persons, not repeating the images of the same person.
    query_image: address to the query image.
    ranking: ranked list containing the information of the images.
    bb: bouding boxes of query image by preprocessing.
    save_dir: directory where are saved the image results.
    """

    data = {'image_path': query_image, 'bounding_boxes': bb.tolist(),
            'ranking': str(ranking)}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
