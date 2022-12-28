# this dummy
# SETUP BATCH DETA
from streamlit_apps.utils import database
import csv
csvFilePath = r'stopwords.csv'

data = []

with open(csvFilePath, encoding='utf-8') as csvf:
    csvReader = csv.DictReader(csvf)
    
    for rows in csvReader:
        
        # key = rows["judul-jurnal"]
        # data[key] = rows
        data.append(rows)
        # data.append(list(rows.values()))

# judul, nama, abstrak, kategori = zip(*data)
# kategori = list(kategori)
# st.write(type(kategori))
import math


def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

if len(data) >= 40:
    step = len(data)/40 # total data bagi 40 (maksimum data satu request DETA) 
    step = math.floor(step)
    output = [data[i:i + step] for i in range(0, len(data), step)]

    #FUNGSI BATCH DETA
    import time
    for trip in output:
        database.stopword.put_many(trip)
        time.sleep(1)

else: print("this program only for 40+ length data")

print("program ends")


# import json
# jsonFilePath = r'dataset.json'

# with open(jsonFilePath, 'w', encoding='utf-8-sig') as jsonf:
#     jsonf.write(json.dumps(output, indent=4))


# # database.data_set.put_many(data)

# # xyz = database.insert_user("usern", "usr", "test")


# # end dummy

# def filter_csv():
#     from streamlit_gallery.utils.lang_detection import language_detection
#     lines = list()
#     # memberName = input("Please enter a member's name to be deleted.")
#     with open('dataset.csv', 'r') as readFile:
#         reader = csv.reader(readFile)
#         for row in reader:
#             lines.append(row)
#             for field in row:
#                 if language_detection(field) != 'id':
#                     lines.remove(row)
#     with open('mycsv.csv', 'w') as writeFile:
#         writer = csv.writer(writeFile)
#         writer.writerows(lines)

# filter_csv()