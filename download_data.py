import wget

if __name__ = '__main__':
    url = 'https://opendata.paris.fr/explore/dataset/paris-wi-fi-utilisation-des-hotspots-paris-wi-fi/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B'
    file = wget.download(url, out='data/hotspots.csv')