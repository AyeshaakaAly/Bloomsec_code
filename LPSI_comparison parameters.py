import hmac
import hashlib
import os
from Crypto.Random import get_random_bytes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Crypto.Cipher import AES
from openpyxl import Workbook
import base64
import time as tm
import math


def extract_keywords_from_file(file_path):
    # Read the content of the file
    try:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()  # Read the entire content of the file
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    # Initialize TfidfVectorizer to extract keywords
    vectorizer = TfidfVectorizer(stop_words='english')

    # Apply the vectorizer to the content
    tfidf_matrix = vectorizer.fit_transform([content])  # The input needs to be a list of documents (even if it's one)

    # Get the feature names (keywords) and their corresponding TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Sum the TF-IDF scores across the document (row)

    # Combine keywords and their scores into a list of tuples
    keyword_scores = list(zip(feature_names, tfidf_scores))

    # Sort the keywords by their TF-IDF score in descending order
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

    # Return the top N keywords
    return sorted_keywords

def encryptData(data, key):
    key = str(key)  # Ensure key is a string
    kbb = key.encode('utf-8')
    kbb = kbb[:32] if len(kbb) > 32 else kbb.ljust(32, b'\0')  # Ensure 32-byte key

    # Ensure the data is a string and then encode it to bytes
    data = str(data)  # Convert data to string if necessary
    data_bytes = data.encode('utf-8', errors='ignore')

    # Generate a random IV (Initialization Vector)
    iv = get_random_bytes(12)  # GCM mode typically uses 12-byte IV

    cipher = AES.new(kbb, AES.MODE_GCM, nonce=iv)

    # Encrypt the data
    encrypted_data, tag = cipher.encrypt_and_digest(data_bytes)

    # Combine the IV, encrypted data, and the tag for storage
    encrypted_data_base64 = base64.b64encode(iv + tag + encrypted_data).decode('utf-8')

    return encrypted_data_base64

def decryptData(encrypted_data_base64, key):
    key = str(key)  # Ensure key is a string
    kbb = key.encode('utf-8')
    kbb = kbb[:32] if len(kbb) > 32 else kbb.ljust(32, b'\0')  # Ensure 32-byte key

    # Decode the base64 encoded encrypted data
    encrypted_data_with_nonce_and_tag = base64.b64decode(encrypted_data_base64)

    # Extract the IV, authentication tag, and encrypted data
    iv = encrypted_data_with_nonce_and_tag[:12]
    tag = encrypted_data_with_nonce_and_tag[12:28]
    encrypted_data = encrypted_data_with_nonce_and_tag[28:]

    # Create the AES cipher in GCM mode using the same IV and tag
    cipher = AES.new(kbb, AES.MODE_GCM, nonce=iv)
    
    # Decrypt the data and verify its authenticity
    decrypted_data = cipher.decrypt_and_verify(encrypted_data, tag)

    # Convert the decrypted data back to a string
    decrypted_data_str = decrypted_data.decode('utf-8', errors='ignore')  # Handle potential decoding issues

    return decrypted_data_str



def timeinMilliSeconds():
    #obj = tm.gmtime(0)
    #epoch = tm.asctime(obj)
    curr_time = round(tm.time()*1000)
    return curr_time




# Path to dataset
def setup_Phase(dataset_path, ks, kw):
    t1 = timeinMilliSeconds()
    if not os.path.exists(dataset_path):
        print(f"Path does not exist: {dataset_path}")
    else:
        print(f"Processing files in: {dataset_path}")

    wb = Workbook()
    sheet = wb.active
    sheet['A1'] = 'path'
    sheet['B1'] = 'x'
    sheet['C1'] = 'e'
    sheet['D1'] = 'keyword'
    i = 2

    keyword_files_map = {}
    tid = 0
    total_files=109789
    total_e_size = 0 # Initialize total size of e
    total_x_size = 0 

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            keywords = extract_keywords_from_file(file_path)
            print(file_path)
            for keyword in keywords:
                if keyword[0] not in keyword_files_map:
                    keyword_files_map[keyword[0]] = ""
                if keyword_files_map[keyword[0]]:
                    keyword_files_map[keyword[0]] += ", "
                tid += 1
                keyword_files_map[keyword[0]] += file_path

    for keyword, files_string in keyword_files_map.items():
        print(f"{keyword} processed")
        hmac_object1 = hmac.new(ks.encode(), str(keyword).encode(), hashlib.sha256)
        x = hmac_object1.hexdigest()
        total_x_size += len(x.encode('utf-8'))  # Add size of x to total_x_size

        hmac_object2 = hmac.new(kw.encode(), str(keyword).encode(), hashlib.sha256)
        kwbar = hmac_object2.hexdigest()
        e = encryptData(files_string, kwbar)
        
        # Calculate the size of e in bytes
        e_size = len(e.encode('utf-8'))  # Size of e in bytes
        total_e_size += e_size  # Add to the total size

        sheet[f'A{i}'] = files_string
        sheet[f'B{i}'] = x
        sheet[f'C{i}'] = str(e)
        sheet[f'D{i}'] = str(keyword)
        i += 1

    wb.save('CloudDataBase.xlsx')
    average_e_size = total_e_size / total_files

    return i - 1, tid, average_e_size, total_x_size
                   

def token_Genration(searchQuery,ks,kw):
    hmac_object1 = hmac.new(kw.encode(), searchQuery.encode(), hashlib.sha256)
    wk = hmac_object1.hexdigest()
    wk_size = len(wk.encode('utf-8'))
    
    hmac_object2 = hmac.new(ks.encode(), searchQuery.encode(), hashlib.sha256)
    wt = hmac_object2.hexdigest()
    wt_size = len(wk.encode('utf-8'))
    return wk,wt,wk_size, wt_size

 

def serach_Query(wt):
    kpri,t,f=LSPI(wt)
    return kpri,t,f
    

def LSPI(y):
    kpri="sdfdsasdflkajsflkjrsfd";
    ksym="rdfdsasdflkajsflkjrsfd";   
    t,f=token_Genration(y,kpri,ksym)
    df = pd.read_excel('CloudDataBase.xlsx', sheet_name='Sheet')  # 'Sheet1' can be replaced with your sheet name

    wb = Workbook()
    sheet = wb.active
    # Write headers for the columns
    sheet['A1'] = 'C'
    sheet['B1'] = 'T'
   

    #print(df['x'])
    for index, row in df.iterrows():
    # Accessing the 'x' and 'e' values from each row
        x_value = str(row['x'])
        e_value = str(row['e'])
        
        hmac_object1 = hmac.new(kpri.encode(), x_value.encode(), hashlib.sha256)
        v = hmac_object1.hexdigest()
        hmac_object2 = hmac.new(ksym.encode(), x_value.encode(), hashlib.sha256)
        sk = hmac_object2.hexdigest()
        epath=encryptData(e_value, sk)
        
        
        
        i = int(v, 16)
        i=i%1048576+1
        print(i)
        sheet[f'A{i}'] = v
        sheet[f'B{i}'] = epath
        
    wb.save('CT.xlsx')              
    print("Data has been saved") 
    return kpri,t,f
        #i=2
    
    
    #print(df['e'])
    #print(t)
    #print(f)

#i=731328
#i=10
#df = pd.read_excel('CT.xlsx', sheet_name='Sheet')  # 'Sheet1' can be replaced with your sheet name
#value_c = df.loc[i-2, 'C']
#print(value_c)


#Main function


# Main function
dataset_path = r"D:\MS\New folder"  # Replace with your dataset path
ks = "AFDDDFFFEG"
kw = "Gdsfafeerf"

t1 = timeinMilliSeconds()
kcount, tid, average_e_size,total_x_size = setup_Phase(dataset_path, ks, kw)  # Now returns average_e_size
t2 = timeinMilliSeconds()
print("Data has been saved")

# Token Generation Phase
t3 = timeinMilliSeconds()
squery = "multexinvestor"
wk, wt,wk_size,wt_size = token_Genration(squery, ks, kw)
t4 = timeinMilliSeconds()
communication_cost = wk_size+total_x_size
token_genration_cost=wk_size+wt_size
Setup_storage=kcount*256+tid/kcount*kcount+average_e_size*kcount

# Search Query Phase
t5 = timeinMilliSeconds()
kpri, t, f = serach_Query(wt)
hmac_object1 = hmac.new(kpri.encode(), wt.encode(), hashlib.sha256)
h_of_y = hmac_object1.hexdigest()
i = int(h_of_y, 16)
i = i % 1048576 + 1
print(i)
df = pd.read_excel('CT.xlsx', sheet_name='Sheet')  # 'Sheet1' can be replaced with your sheet name
value_c = str(df.loc[i - 2, 'C'])
value_t = str(df.loc[i - 2, 'T'])
print(value_c)
print(value_t)
if value_c == f:
    print("Found")
    ddata = decryptData(value_t, str(t))
    path = decryptData(ddata, str(wk))
    print(path)
else:
    print("Not Found")
t6 = timeinMilliSeconds()

print(f"Total file identifier save: {tid}")
print(f"Total keyword: {kcount}")
print(f"Average file identifier w.r.t each keyword {tid / kcount}")
print(f"Average size of e per file: {average_e_size} bytes")  # Print average size of e
print(f"Total setup Time in milliseconds: {t2 - t1}")
print(f"Token Generation Time in milliseconds: {t4 - t3}")
print(f"Search Query Time in milliseconds: {t6 - t5}")
print(f"Setup phase Communication cost: {communication_cost} bytes") 
print(f"token generation phase Communication cost: {token_genration_cost} bytes") 
print(f"Setup phase Storage cost: {Setup_storage} bytes")