from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

"""
Basic setting for fastapi backend.
"""
origins=[
   'http//localhost:3000',
   'localhost:3000'
]


app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=['*'],
   allow_headers=['*']
)


# get router
@app.get('/',tags=['root'])
async def read_root()->dict:
   return {'message':'welcome to fastapi!'}

