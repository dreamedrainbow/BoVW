import flickr
import urllib, urlparse 
import os
import sys

DATA_DIR = 'data'

if len(sys.argv)>1: 
    tag = sys.argv[1]
else:
    print 'no tag specified'

def main():
    # downloading image data
    f = flickr.photos_search(tags=tag)
    # create a directory for the files if it doesn't exist
    path = os.path.join(DATA_DIR, tag)
    if not os.path.isdir(path):
        os.makedirs(path)
    os.chdir(path)
    urllist = [] #store a list of what was downloaded
    # downloading images 
    for k in f:
        url = k.getURL(size='Medium', urlType='source') 
        urllist.append(url)
        image = urllib.URLopener()
        image.retrieve(url, os.path.basename(urlparse.urlparse(url).path)) 
        print 'downloading:', url

    # write the list of urls to file 
    fl = open('urllist.txt', 'w') 
    for url in urllist:
        fl.write(url+'\n') 
    fl.close()

if __name__ == "__main__":
    main()