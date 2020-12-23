#!/usr/bin/env python3

import json
import os
import sys
import asyncio
import pathlib
import websockets
import concurrent.futures
import logging
from dotenv import load_dotenv
from classifier import init, predict_phrase

'''
Init and configuration
'''
# Enable loging
logging.root.handlers = []
logging.basicConfig(
    encoding='utf-8',
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('text-assessment-server')

# Get config
load_dotenv(verbose=True)
server_interface = os.environ.get('SERVER_INTERFACE', '0.0.0.0')
server_port = int(os.environ.get('SERVER_PORT', 2700))

# Init asyncio
pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))
loop = asyncio.get_event_loop()

# Init classifier
freqs, theta = init()

'''
Handler for messages
'''
def classify_message(message):
    if isinstance(message, str):
        # parse request
        request = json.loads(message)

        # classify phrase
        predicted_sentiment = predict_phrase(request['sentence'], freqs, theta)

        # prepare and return response with the original phrase and its sentiment
        response = {
            'key': request['key'],
            'sentence': request['sentence'],
            'sentiment': predicted_sentiment[0][0]
        }
        return json.dumps(response)

'''
Main handler for incoming client socket connections
'''
async def handler(websocket, path):
    while True:
        # retrieve request
        message = await websocket.recv()
        logger.info('Request: ' + message)

        # run processing in a thread
        response = await loop.run_in_executor(pool, classify_message, message)
        logger.info('Response: ' + response)

        # send back response
        await websocket.send(response)

'''
Init and start socket server
'''
logger.info("Starting server on host " + server_interface + ", port " + str(server_port))
start_server = websockets.serve(
    handler, server_interface, server_port)
loop.run_until_complete(start_server)
loop.run_forever()

