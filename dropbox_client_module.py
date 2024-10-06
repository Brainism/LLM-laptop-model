import os
import dropbox

DROPBOX_API_TOKEN = os.getenv("DROPBOX_API_TOKEN")

if not DROPBOX_API_TOKEN:
    raise ValueError("Dropbox API 토큰이 환경 변수에 설정되지 않았습니다.")

dbx = dropbox.Dropbox(DROPBOX_API_TOKEN)

def get_account_info():
    try:
        account_info = dbx.users_get_current_account()
        return account_info
    except dropbox.exceptions.AuthError as err:
        print("인증 오류: 액세스 토큰을 확인하세요.")
        return None
