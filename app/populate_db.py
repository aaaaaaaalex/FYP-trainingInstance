import mysql.connector

db = mysql.connector.connect(
    host="db",
    user="root",
    passwd="test",
    database='app'
)

cursor = db.cursor()

def populate_imgs():
    URLS_path = "./dataset/url/{}/url.txt"
    with open('./dataset/classes.config') as fd:
        for cl in fd:
            classname = cl.strip('\n')
            print(classname)
            cursor.execute("""
                INSERT INTO `Category`
                    (`cID`, `cName`) 
                    VALUES (%s, %s);
                """, (None, classname))

            classID = cursor.lastrowid
            print(classID)

            with open (URLS_path.format(classname)) as urls:
                for url in urls:
                    if url != "":
                        fmt_url = url.strip("\n\r")
                        cursor.execute("""
                            INSERT INTO `Img`
                                (`imgID`, `imgURL`, `userID`)
                                VALUES (%s, %s, %s)
                            """, (None, fmt_url, None))

                        imgID = cursor.lastrowid

                        cursor.execute("""
                            INSERT INTO `ImageCategory`
                                (icID, cID, imgID)
                                VALUES (%s, %s, %s)
                        """, (None, classID, imgID) )


        db.commit()

populate_imgs()