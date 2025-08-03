import asyncio
from storage.s3_client import S3Client

async def test():
    client = S3Client()
    health = await client.health_check()
    print(f"S3 Connection: {'OK' if health else 'Failed'}")

asyncio.run(test())
