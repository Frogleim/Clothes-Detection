import psycopg2
import bcrypt


def db_connect():
    conn = psycopg2.connect(host="localhost",
                            database="postgres",
                            user="postgres",
                            password="123456"
                            )

    return conn


def create_table():
    with db_connect() as conn:
        cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS lookat_db ("
                   "user_id int,"
                   "downloaded_image text,"
                   "predicted_image text)",

                   )
    conn.commit()


def save_users(id, image):
    connection = db_connect()
    curs = connection.cursor()
    curs.execute("INSERT INTO lookat  (user_id, downloaded_img) VALUES (%s, %s)",
                 (id, image))
    connection.commit()
    connection.close()
    print('Saved Successfully')


def add_data(data, user_id, count=None):
    with db_connect() as conn:
        cursor = conn.cursor()
        if count is None:
            cursor.execute("ALTER TABLE lookat ADD COLUMN predicted bytea",)
            print("Table created....")
            cursor.execute("INSERT INTO lookat  (users_id, predicted) VALUES (%s, %s)",
                         (user_id, data))
            print("Saved...")
            conn.commit()
        else:
            for imgs in range(count):
                print(imgs)
                cursor.execute(f"ALTER TABLE lookat ADD COLUMN predicted_{imgs} bytea", )
                # print("Table created....")
                # cursor.execute(f"INSERT INTO lookat  (users_id, predicted_{imgs}) VALUES (%s, %s)",
                #                (user_id, data[imgs].tobytes()))
                # print("Saved...")
                conn.commit()


def getitem(user_id):
    with db_connect() as conn:
        cursor = conn.cursor()
    cursor.execute("SELECT downloaded_img FROM lookat WHERE user_id = %s",
                   (str(user_id))
                   )
    result = cursor.fetchone()
    conn.commit()
    return result


if __name__ == '__main__':
    res = getitem(1)
    print(res)
