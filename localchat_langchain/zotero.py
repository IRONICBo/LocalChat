import os
from pyzotero import zotero

# From env
user_id = os.getenv("ZOTERO_USER_ID")

zot = zotero.Zotero(user_id, 'user', local=True)
a = zot.items()
print(a)