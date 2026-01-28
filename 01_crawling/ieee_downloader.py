import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def login_and_access_ieee(username, password, target_url):
    """
    도서관 로그인 후 IEEE 페이지 접속
    Phase 1: 도서관 로그인
    Phase 2: IEEE 페이지 접근
    """
    print(">>> [Phase 1] 도서관 로그인 중...")
    
    # Selenium 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # 도서관 로그인 페이지 접속
        login_url = "https://oasis.ssu.ac.kr/login"
        print(f">>> 로그인 페이지 접속: {login_url}")
        driver.get(login_url)
        wait = WebDriverWait(driver, 30)
        
        # 로그인 필드 찾기 및 입력
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "input")))
            inputs = driver.find_elements(By.TAG_NAME, "input")
            
            username_field = None
            password_field = None
            
            for inp in inputs:
                if inp.is_displayed():
                    if not username_field and inp.get_attribute('type') in ['text', 'email', '']:
                        username_field = inp
                    elif inp.get_attribute('type') == 'password':
                        password_field = inp
                        break
            
            if not username_field or not password_field:
                username_field = driver.find_element(By.CSS_SELECTOR, "input[id*='id']")
                password_field = driver.find_element(By.CSS_SELECTOR, "input[type='password']")

            print(">>> 로그인 정보 입력 중...")
            username_field.clear()
            username_field.send_keys(username)
            password_field.clear()
            password_field.send_keys(password)
            password_field.send_keys(Keys.RETURN)
            
            print(">>> 로그인 완료 대기 중...")
            # 로그인 완료 확인 (URL이 변경될 때까지 대기)
            try:
                wait.until(lambda d: d.current_url != login_url)
                time.sleep(3)  # 추가 안정화 시간
                print(f">>> 로그인 후 현재 URL: {driver.current_url}")
            except:
                print(">>> 로그인 완료 확인 중 타임아웃 (계속 진행)")
                time.sleep(3)
            
        except Exception as e:
            print(f">>> 로그인 오류 (이미 로그인되어 있을 수 있음): {e}")
        
        # Phase 2: IEEE 페이지 접근
        print(f">>> [Phase 2] IEEE 페이지 접근 중: {target_url}")
        driver.get(target_url)
        
        # 페이지가 완전히 로드될 때까지 대기
        print(">>> 페이지 로딩 대기 중...")
        try:
            # IEEE 페이지의 body 요소가 나타날 때까지 대기
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(5)  # 추가 로딩 시간
            
            # iframe이 있으면 iframe 로딩 대기
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
                print(">>> iframe 로딩 완료")
                time.sleep(3)
            except:
                print(">>> iframe 없음 또는 로딩 대기 중 타임아웃")
            
        except Exception as e:
            print(f">>> 페이지 로딩 대기 중 오류: {e}")
            time.sleep(5)  # 최소한의 대기 시간
        
        current_url = driver.current_url
        print(f">>> 현재 URL: {current_url}")
        
        # IEEE 페이지 접근 확인
        if "oasis.ssu.ac.kr" in current_url and "ieee" not in current_url:
            print(">>> FAILED: 도서관 홈으로 리다이렉트됨. 로그인 세션이 유효하지 않을 수 있습니다.")
            print(f">>> 예상 URL: {target_url}")
            print(f">>> 실제 URL: {current_url}")
            return None
        
        # 페이지 제목 확인
        try:
            page_title = driver.title
            print(f">>> 페이지 제목: {page_title}")
        except:
            pass
        
        print(">>> SUCCESS: IEEE 페이지 접속 완료!")
        return driver
        
    except Exception as e:
        print(f">>> ERROR: 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    u_id = "20211737"
    u_pw = "Ajdcjddl12@"
    target_pdf_url = "https://ieeexplore-ieee-org-ssl.openlink.ssu.ac.kr/stamp/stamp.jsp?tp=&arnumber=10758606"
    
    driver = login_and_access_ieee(u_id, u_pw, target_pdf_url)
    
    if driver:
        print(">>> 브라우저를 열어둡니다. 종료하려면 Enter를 누르세요...")
        input()
        driver.quit()
    else:
        print(">>> 실패: IEEE 페이지 접속에 실패했습니다.")

