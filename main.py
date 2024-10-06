import os
from dropbox_client_module import dbx, get_account_info

def main():
    account_info = get_account_info()
    if account_info:
        print(f"사용자 이름: {account_info.name.display_name}")
    else:
        print("Dropbox 계정 정보를 가져오는데 실패했습니다.")

if __name__ == "__main__":
    main()
