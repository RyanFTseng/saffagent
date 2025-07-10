from dotenv import load_dotenv
import lancedb
from typing import List
from interface.base_datastore import BaseDatastore, DataItem
from lancedb.table import Table
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

class Datastore:
    DB_PATH = "data/sample-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        aiplatform.init(project="746472204967", location="us-west1")
        self.vector_dimensions = 3072
        self.embedding_model = GenerativeModel("text-embedding-004")
        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table: Table = self._get_table()

    def reset(self) -> Table:
        # Drop the table if it exists
        try:
            self.vector_db.drop_table(self.DB_TABLE_NAME)
        except Exception as e:
            print("Unable to drop table. Assuming it doesn't exist.")
