
import fastapi
import types
import typing
import pydantic
import PIL.Image
import requests
import torch
import uvicorn

import advanced

application = fastapi.FastAPI()

@application.get("/status/")
async def status():
  message = 'Server is working now.'
  return(message)

class Packet(pydantic.BaseModel):
  image: typing.Union[str, None] = 'https://www1.cgmh.org.tw/adrlnk/images/2-3.jpg'
  threshold: float = 0.5
  score: typing.Union[float, None] = None
  prediction: typing.Union[int, None] = None
  pass

model = advanced.model.downstreamModel(weight={"backbone":"MobileNet_V2_Weights.IMAGENET1K_V1", "senior":None})
model.loadWeight("./static/Fined-Tuning-MobileNet_V2/pretrained_weight.pickle")
transform = advanced.model.defineTransform(inference=True)
loadPicture = lambda url: PIL.Image.open(requests.get(url, stream=True).raw)
runSoftmax = lambda tensor: torch.nn.Softmax(1)(tensor)

@application.post("/classifier/")
async def classifier(packet: Packet):
  url = packet.image
  picture = loadPicture(url)
  picture = transform(picture).unsqueeze(0)
  inference = model(picture)
  _, score = runSoftmax(inference).squeeze(0).tolist()
  prediction = "higher" if(score>packet.threshold) else 'lower'
  pass
  packet.score = score
  packet.prediction = prediction
  response = packet.dict()
  return(response)

if(__name__=='__main__'):
  uvicorn.run("server:application", host="0.0.0.0", port=65081)
  pass