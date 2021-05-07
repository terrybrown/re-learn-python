import asyncio


async def countAsync():
    print("One")
    await asyncio.sleep(1)
    print("Two")

def count():
    print("One")
    time.sleep(1)
    print("Two")

async def mainAsync():
    await asyncio.gather(countAsync(), countAsync(), countAsync())

def main():
    for _ in range(3):
        count()

# The async/await operation here should return in a fraction of the time as they all run in parallel
if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(mainAsync())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

    s = time.perf_counter()
    main()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

