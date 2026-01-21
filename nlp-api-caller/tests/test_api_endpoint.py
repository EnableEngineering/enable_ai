#!/usr/bin/env python3
"""Quick test to check API endpoint availability."""

import requests
import os

def test_api():
    token_url = 'http://localhost:8002/api/token/'
    print(f'Testing authentication at: {token_url}')
    
    email = os.getenv('API_EMAIL', 'admin@example.com')
    password = os.getenv('API_PASSWORD', 'adminpassword')
    
    try:
        # Test authentication
        response = requests.post(token_url, json={'email': email, 'password': password}, timeout=5)
        print(f'Auth status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('access') or data.get('access_token') or data.get('token')
            print(f'Token obtained: {token[:20] if token else "None"}...')
            
            # Test users endpoint
            users_url = 'http://localhost:8002/api/users/'
            print(f'\nTesting: {users_url}')
            headers = {'Authorization': f'Bearer {token}'}
            response2 = requests.get(users_url, headers=headers, timeout=5)
            print(f'Status: {response2.status_code}')
            
            if response2.status_code == 200:
                print('✓ Users endpoint works!')
                result = response2.json()
                print(f'Result keys: {list(result.keys()) if isinstance(result, dict) else "list"}')
                
                # Check if this is a navigation page
                if isinstance(result, dict) and 'users' in result:
                    print('\nThis appears to be a navigation/index page.')
                    print('Trying the actual users list endpoint...')
                    
                    # Try the users list endpoint
                    users_list_url = result.get('users')
                    if users_list_url:
                        response3 = requests.get(users_list_url, headers=headers, timeout=5)
                        print(f'\nUsers list URL: {users_list_url}')
                        print(f'Status: {response3.status_code}')
                        
                        if response3.status_code == 200:
                            users_data = response3.json()
                            print(f'\n✓ Got actual users data!')
                            
                            if isinstance(users_data, dict):
                                print(f'Response keys: {list(users_data.keys())}')
                                if 'results' in users_data:
                                    print(f'Number of users: {len(users_data["results"])}')
                                    if users_data['results']:
                                        print(f'First user sample: {users_data["results"][0]}')
                            elif isinstance(users_data, list):
                                print(f'Number of users: {len(users_data)}')
                                if users_data:
                                    print(f'First user sample: {users_data[0]}')
            else:
                print(f'✗ Error response:')
                print(response2.text[:300])
        else:
            print(f'Auth failed: {response.text[:200]}')
            
    except requests.exceptions.ConnectionError:
        print('✗ Connection refused - API server not running on port 8002')
    except Exception as e:
        print(f'✗ Error: {e}')

if __name__ == '__main__':
    test_api()
