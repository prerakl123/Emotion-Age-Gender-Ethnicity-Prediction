import sqlite3


class DB:
    def __init__(self, db_file: str = 'mapping.db'):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.__init_tables__()

    def __init_tables__(self):
        self.cursor.executescript(
"""
CREATE TABLE IF NOT EXISTS person(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NULL,
    db_hash VARCHAR(255) NOT NULL
);
"""
        )
        self.conn.commit()

    def set(self, stmt: str, *args):
        print(args)
        self.cursor.execute(stmt, args)
        self.conn.commit()

    def get(self, stmt: str, *args):
        print(args)
        self.cursor.execute(stmt, args)
        result = self.cursor.fetchall()[0]
        return result

    def set_name(self, db_hash, name=None):
        self.set("INSERT INTO person(name, db_hash) VALUES(?, ?);", name, db_hash)

    def get_name(self, db_hash):
        return self.get("SELECT name FROM person WHERE db_hash=?;", db_hash)


if __name__ == '__main__':
    db = DB()
    db.set_name("2390ruyidfohc23908rhifsn", "Prerak")
