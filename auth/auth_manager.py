import sqlite3
import os
import bcrypt


# DB 파일 위치 설정
DB_DIR = "db"
DB_PATH = os.path.join(DB_DIR, "users.db")


# DB 초기화 함수 (폴더 & 테이블 자동 생성)
def init_db():
    # db 폴더가 없으면 생성
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    # SQLite 연결
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # USERS 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS USERS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# 비밀번호 해쉬 생성 함수
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


# 비밀번호 검증 함수
def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except:
        return False


# 회원가입 함수
def create_user(username: str, password: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        password_hash = hash_password(password)

        cursor.execute("""
            INSERT INTO USERS (username, password_hash)
            VALUES (?, ?)
        """, (username, password_hash))

        conn.commit()
        conn.close()
        return True

    except sqlite3.IntegrityError:
        # username 중복
        return False


# 로그인 검증 함수
def verify_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT password_hash FROM USERS WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False  # 존재하지 않는 유저

    password_hash = row[0]
    return verify_password(password, password_hash)


# 회원탈퇴 함수
def delete_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM USERS WHERE username = ?", (username,))
    conn.commit()
    conn.close()

    # 유저 데이터 폴더도 삭제할 수 있도록 True 반환
    return True


# 모듈 실행 시 DB 자동 초기화
if __name__ == "__main__":
    init_db()
    print("SQLite DB 초기화 완료!")
