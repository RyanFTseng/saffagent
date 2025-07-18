import requests

def send_post_request(system_message, user_message):
    url = "https://3l4gubpw4sp3lfijdhmwi3x6ue0jzkik.lambda-url.us-west-1.on.aws/agent"
    data = {"message": system_message + " " + user_message}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print(f"Error: {e}")
