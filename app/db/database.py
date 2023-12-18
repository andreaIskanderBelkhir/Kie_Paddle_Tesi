from app.config import cfg
from app.db.database_helper import dataset_helper, item_helper
from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

print("Init database")
client = AsyncIOMotorClient(cfg.DATABASE_URL)
database = client[cfg.DATABASE_NAME]

dataset_collection = database.get_collection("dataset")
item_collection = database.get_collection("item")


# dataset - crud operations







# Retrieve all dataset present in the database
async def retrieve_datasets():
    datasets = []
    async for dataset in dataset_collection.find():
        print(dataset)
        datasets.append(dataset_helper(dataset))
    return datasets


# Add a new dataset into to the database
async def add_dataset(dataset_data: dict) -> dict:
    dataset = await dataset_collection.insert_one(dataset_data)
    new_dataset = await dataset_collection.find_one({"_id": dataset.inserted_id})
    return dataset_helper(new_dataset)


# Retrieve a dataset with a matching ID
async def retrieve_dataset(id: str) -> dict:
    dataset = await dataset_collection.find_one({"_id": ObjectId(id)})
    if dataset:
        return dataset_helper(dataset)


# Update a dataset with a matching ID
async def update_dataset(id: str, data: dict) -> bool:
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    dataset = await dataset_collection.find_one({"_id": ObjectId(id)})
    if dataset:
        updated_dataset = await dataset_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_dataset:
            return True
        return False


# Delete a dataset from the database
async def delete_dataset(id: str) -> bool:
    dataset = await dataset_collection.find_one({"_id": ObjectId(id)})
    if dataset:
        await dataset_collection.delete_one({"_id": ObjectId(id)})
        return True


# item - crud operations


# Retrieve all item present in the database
async def retrieve_items():
    items = []
    async for item in item_collection.find():
        items.append(item(item))
    return items


# Add a new item into to the database
async def add_item(item_data: dict) -> dict:
    item = await item_collection.insert_one(item_data)
    new_item = await item_collection.find_one({"_id": item.inserted_id})
    return new_item


# Retrieve a item with a matching ID
async def retrieve_item(id: str) -> dict:
    item = await item_collection.find_one({"_id": ObjectId(id)})
    if item:
        return item_helper(item)


# Update a item with a matching ID
async def update_item(id: str, data: dict):
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    item = await item_collection.find_one({"_id": ObjectId(id)})
    if item:
        updated_item = await item_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_item:
            return True
        return False


# Delete a item from the database
async def delete_item(id: str):
    item = await item_collection.find_one({"_id": ObjectId(id)})
    if item:
        await item_collection.delete_one({"_id": ObjectId(id)})
        return True
