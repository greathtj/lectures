from roboflow import Roboflow
rf = Roboflow(api_key="W6heiWaOfMtbe0JEgZzt")
project = rf.workspace().project("face-jogph")
model = project.version(2).model

# infer on a local image
# print(model.predict("1st_test.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("3persons.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())