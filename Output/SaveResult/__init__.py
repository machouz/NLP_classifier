import datetime
import io


def toCSV(model, features, added_features, f1, path):
    now = datetime.datetime.today()
    hour, minute, seconds = now.hour, now.minute, now.second,
    added_features = [item for sublist in added_features if sublist for item in sublist]
    with open(path, 'a+') as file:
        file.seek(0)
        if not file.read(1):
            file.write('{},{},{},{},{}\n'.format("Time", "model", "Created features",
                                                 "Received features", "Macro F1"))
        file.seek(0, io.SEEK_END)
        file.write('{}:{}:{}, {}, {}, {}, {}\n'.format(hour, minute, seconds, model, ' '.join(features),
                                                       ' '.join(added_features), f1))
