import pymongo

HOST = "mongodb+srv://pinku03260707:<password>@kwargs.za1zguv.mongodb.net/"

with pymongo.MongoClient(host = HOST) as client:
    for db in client.list_databases():
        print(db)