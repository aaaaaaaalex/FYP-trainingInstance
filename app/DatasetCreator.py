import logging
import pandas as pd
import requests

from io import BytesIO
from os import path, makedirs
from PIL import Image as pimage, ImageFile


class DatasetCreator ():

    def __init__(self, dbcursor, workdir, class_config=None, unskew_data=True):
        self.class_config = class_config
        self.dbcursor = dbcursor
        self.unskew_data = unskew_data
        self.workdir = workdir
        self.dataset_dataframe = None
        
        

    # ensure that there is the same number of data for every class in a dataframe
    @staticmethod
    def __unskew_data__ (df, min_images_per_class=None):

        # find the class with the smallest number of images
        lowest_images = -1
        lowest_classname = ""
        for classname in df['label'].unique():
            class_count = len(df[ df['label'] == classname])
            if (lowest_images < 0 or class_count < lowest_images): 
                lowest_images = class_count
                lowest_classname = classname

        # drop images of other classes to un-skew the dataset
        all_classes = []
        logging.info("\nClass '{}' has the least pulled-images. Image count: {}".format(lowest_classname, lowest_images))
        for classname in df['label'].unique():
            all_imgs_for_class = df[df['label'] == classname]
            redundancy = len(all_imgs_for_class) - lowest_images
            lost_images = redundancy / len(all_imgs_for_class)

            if redundancy > 0 :
                all_imgs_for_class = all_imgs_for_class[:-redundancy]

            if min_images_per_class==None or len(all_imgs_for_class) >= min_images_per_class:
                all_classes.append(all_imgs_for_class)

            #print("{}: {} redundant images - {}% lost.".format(classname, redundancy, lost_images*100))
        
        df = pd.concat(all_classes)
        return df

    
    # attempts to validate if a collection of bytes represents an image by parsing them with PIL.Image
    # throws an exception if the data cannot be parsed
    @staticmethod
    def __sanitise_image__(bytes_data):
        buf = BytesIO(bytes_data)
        img = pimage.open(buf)

        # move data from original image into new image, dropping redundant metadata to save space / avoid warnings
        buf_no_exif = BytesIO()
        img_no_exif = pimage.new(img.mode, img.size)

        img_no_exif.putdata( list(img.getdata()) )
        img_no_exif.save(buf_no_exif, format="jpeg")

        # drop all EXIF data as it is all redundant
        return buf_no_exif.getvalue()


    # downloads images at the URLs provided
    @staticmethod
    def __download_imgs__(links, save_dir):
        # throw error if links isnt a list
        assert type(links) is list
        images_pulled = []

        if not path.exists(save_dir):
            makedirs(save_dir)
        
        # iterate over each link, carrying both the link and it's list index
        for i, link in enumerate(links):
            try:
                # make a GET request and dont follow redirects - timeout after 3 secs
                r = requests.get(link[0], timeout=3)
                r.raise_for_status()

                # make sure the response is an image, not HTML
                img = DatasetCreator.__sanitise_image__(r.content)
                filename = "{}img-{}.jpg".format(save_dir, len(images_pulled))
                with open(filename, 'wb') as f:
                    f.write(img)

                images_pulled.append(filename)

            except (requests.exceptions.RequestException, OSError) as err:
                continue
        
        return images_pulled


    # fetch links to images of the specified classname
    # select imgURL from Img where imgID in (   select imgID from ImageCategory   where cID in (     select cID from Category     where cName = "street"   ) ) limit 10;
    def __get_class_links__(self, classname):
        cursor = self.dbcursor
        cursor.execute("""
            SELECT i.imgURL
            FROM Img i, ImageCategory ic, Category c
            WHERE i.imgID = ic.imgID
                  and ic.cID = c.cID
                  and c.cName = '{}';
            """.format(classname) )

        result = cursor.fetchall()
        return result


    # for every classname in the config path, pull image links
    # and store them in the filesystem, return a table of labels for each file
    def __make_dataset__(self, class_config):
        dataset = {'id': [], 'label': []}

        # begin pulling images for each class
        for classname in class_config:
            if classname == '':
                continue
            
            logging.info("Pulling images for class: {}".format(classname))
            dest_path = '{}dataset/{}/'.format(self.workdir, classname)
            
            # get image links by class
            img_links = self.__get_class_links__(classname)

            # download images using links
            imgs_pulled = DatasetCreator.__download_imgs__(img_links, dest_path)
            for filename in imgs_pulled:
                dataset['id'].append(filename)
                dataset['label'].append(classname)
            logging.debug("'{}' images pulled: {}".format(classname, len(imgs_pulled)))

        # construct a table containing filenames and their corresponding classes
        df = pd.DataFrame(data=dataset)
        df.to_csv('{}dataset/dataset_cache.csv'.format(self.workdir))

        return df


    # get a dataframe of pulled-images
    # triggers a download if it hasn't already pulled images
    def get_dataset_dataframe(self):
        if self.dataset_dataframe is None:

            try:
                if self.class_config:
                    df = self.__make_dataset__(self.class_config)
                # check for a map of cached-images already on disk
                else:
                    df = pd.read_csv('{}dataset/dataset_cache.csv'.format(self.workdir), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError('Dataset Creator was not passed a class config, and no dataset cache is present')

            if self.unskew_data is True:
                df = DatasetCreator.__unskew_data__(df)

            # Give absolute paths for images
            df['id'] = df['id'].apply(
                lambda val:
                    path.abspath(val)
                )
                
            self.dataset_dataframe = df
        
        return self.dataset_dataframe