import redis


r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    password=''
)
r.mset({"Croatia": "Zagreb", "Bahamas": "Nassau"})
print(r.get("Bahamas"))
print(r.exists("Bahamas2"))