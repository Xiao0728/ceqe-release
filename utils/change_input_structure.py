def listDicts_to_keysValuesLists(listDicts, keys):
    output = [[] for k in keys]

    for d in listDicts:
        for k in range(len(keys)):
            output[k].append(d[keys[k]])
    return output