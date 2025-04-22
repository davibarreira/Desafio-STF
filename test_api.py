import asyncio
import aiohttp


async def make_request(session, id):
    async with session.post(
        f"http://localhost:8000/api/pecas/{id}", json={"texto": "Texto de exemplo"}
    ) as response:
        return await response.json()


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        print(results)


asyncio.run(main())
