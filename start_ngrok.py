import subprocess
import requests
import time

def start_ngrok():
    try:
        # ngrok을 백그라운드에서 실행
        process = subprocess.Popen(['/data/ephemeral/ngrok', 'http', '5000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ngrok이 포트 5000에서 실행 중입니다.")
        return process
    except Exception as e:
        print(f"ngrok 실행 중 오류 발생: {e}")

def get_ngrok_url():
    # ngrok 웹 인터페이스에서 URL 정보를 가져오기 위한 대기 시간
    time.sleep(2)  # ngrok이 완전히 실행되기 전에 바로 요청하면 실패할 수 있음
    
    try:
        # ngrok API에서 URL 정보 가져오기
        response = requests.get('http://127.0.0.1:4040/api/tunnels')
        data = response.json()
        public_url = data['tunnels'][0]['public_url']
        print(f"ngrok URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"ngrok URL을 가져오는 중 오류 발생: {e}")
        return None

def main():
    # ngrok 실행
    ngrok_process = start_ngrok()

    # ngrok이 실행된 후 URL 확인
    ngrok_url = get_ngrok_url()
    if ngrok_url:
        print(f"ngrok이 제공하는 URL: {ngrok_url}")
    
    try:
        # ngrok 프로세스를 대기 (강제 종료되지 않도록)
        ngrok_process.wait()
    except KeyboardInterrupt:
        print("ngrok 종료 중...")
        ngrok_process.terminate()

if __name__ == '__main__':
    main()
