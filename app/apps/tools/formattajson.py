import math
import os
import json

def oldjson(result,newresult):
    for idx,x in enumerate(result):
        keylist=[]
        valuelist=[]
        keylist.append(x[0]["pred_id"])
        valuelist.append(x[1]["pred_id"])
        #idx è ogni relazione devo andare in x[idx] e fare enumerate ancora
        newresult["key_value"].append({
            "key":keylist,
            "value":valuelist,
             })


def formatta_annotation(resp_padl):
        i=0
        response=[]
        
        for res in resp_padl:
            bbox=[]
            for indici in res[0]:
                bbox.append({
                    "x":indici[0],
                    "y":indici[1],
                     })
            response.append({
                "text":res[1][0],
                "entity": None,
                "bounding_box":bbox,
                "bounding_box_ocr":None,
                "id_line":i,
                "points": res[0],
                "polygon":[res[0][0][0],res[0][0][1],res[0][1][0],res[0][1][1],res[0][2][0],res[0][2][1],res[0][3][0],res[0][3][1]]
                })

            i=i+1
        return response


def dbjson(items,result):
	visti={
		"vistiK":[],
		"vistiV":[],
		}
	for idx,x in enumerate(result):
		polyx=[x[0]["points"][0][0],x[0]["points"][0][1],x[0]["points"][1][0],x[0]["points"][1][1],x[0]["points"][2][0],x[0]["points"][2][1],x[0]["points"][3][0],x[0]["points"][3][1]]
		polyy=[x[1]["points"][0][0],x[1]["points"][0][1],x[1]["points"][1][0],x[1]["points"][1][1],x[1]["points"][2][0],x[1]["points"][2][1],x[1]["points"][3][0],x[1]["points"][3][1]]

		idk=None
		idv=None

		for ident,ent in enumerate(items["annotations"]):

			if(ent["polygon"]==polyx):
				idk=ent["id_line"]
				if(idk not in visti["vistiK"]):
					visti["vistiK"].append(idk)
					items["entities"].append({
						"type":"Key",
						"ids":idk,
						"text":x[0]["transcription"],
						})
			elif(ent["polygon"]==polyy):
				idv=ent["id_line"]
				if(idv not in visti["vistiV"]):
					visti["vistiV"].append(idv)
					items["entities"].append({
						"type":"Value",
						"ids":idv,
						"text":x[1]["transcription"],
						})
			elif(idk is not None and idv is not None):
				break

		items["relations"].append({
			"key":[idk],
			"value":[idv]
			})


	for idp,p in enumerate(items["annotations"]):
		p.pop("points")
		p.pop("polygon")
	return items

# idea di base prendo i singoli json creati dalla inferenza e li passo a quello grosso che contiene annche il resto con info db
def additems(item):
	with open(
		os.path.join("./app/save_re/feedbackjson.json"),"r",encoding ="utf-8"
		) as fout:
		jsondb=json.load(fout)
		fout.close()

	jsondb["id_docs"].append(item["id"])
	jsondb["items"].append(item)
	with open(
	os.path.join("./app/save_re/feedbackjson.json"),"w",encoding ="utf-8"
		) as fout:
		json.dump(jsondb,fout)
		fout.close()




