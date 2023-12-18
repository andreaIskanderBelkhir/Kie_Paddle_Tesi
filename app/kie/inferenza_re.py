import os 
import cv2
import json
import paddle
import paddle.distributed as dist
from paddleocr import ppocr

from ppocr.data import create_operators, transform
from app.apps.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.visual import draw_re_results
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps, print_dict
import app.kie.inferenza_ser as SerPredictor
import yaml
import numpy as np


folderRe="./app/kie/md_re"

configT="./app/kie/re_conf.yml"


def make_input(ser_inputs, ser_results):
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}
    batch_size, max_seq_len = ser_inputs[0].shape[:2]
    entities = ser_inputs[8][0]
    ser_results = ser_results[0]
    assert len(entities) == len(ser_results)

    # entities
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_results, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])

    entities = np.full([max_seq_len + 1, 3], fill_value=-1, dtype=np.int64)
    entities[0, 0] = len(start)
    entities[1:len(start) + 1, 0] = start
    entities[0, 1] = len(end)
    entities[1:len(end) + 1, 1] = end
    entities[0, 2] = len(label)
    entities[1:len(label) + 1, 2] = label

    # relations
    head = []
    tail = []
    for i in range(len(label)):
        for j in range(len(label)):
            if label[i] == 1 and label[j] == 2:
                head.append(i)
                tail.append(j)

    relations = np.full([len(head) + 1, 2], fill_value=-1, dtype=np.int64)
    relations[0, 0] = len(head)
    relations[1:len(head) + 1, 0] = head
    relations[0, 1] = len(tail)
    relations[1:len(tail) + 1, 1] = tail

    entities = np.expand_dims(entities, axis=0)
    entities = np.repeat(entities, batch_size, axis=0)
    relations = np.expand_dims(relations, axis=0)
    relations = np.repeat(relations, batch_size, axis=0)

    # remove ocr_info segment_offset_id and label in ser input
    if isinstance(ser_inputs[0], paddle.Tensor):
        entities = paddle.to_tensor(entities)
        relations = paddle.to_tensor(relations)
    ser_inputs = ser_inputs[:5] + [entities, relations]

    entity_idx_dict_batch = []
    for b in range(batch_size):
        entity_idx_dict_batch.append(entity_idx_dict)
    return ser_inputs, entity_idx_dict_batch




class Inferenza_re(object):

    def __init__(self):

        config = yaml.load(open(configT, 'rb'),Loader=yaml.Loader)

        global_config = config['Global']

        self.ser_engine = SerPredictor.SerPredictor()		
        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],global_config)
        # build model
        self.model = build_model(config['Architecture'])

        load_model(config, self.model, model_type=config['Architecture']["model_type"])

        self.model.eval()

    def __call__(self, data):
        ser_results, ser_inputs = self.ser_engine(data)
        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        if self.model.backbone.use_visual_backbone is False:
            re_input.pop(4)
        preds = self.model(re_input)
        post_result = self.post_process_class(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)
        return post_result





if __name__ == '__main__':
    
    os.makedirs("./save_re", exist_ok=True)

    ser_re_engine = Inferenza_re()



    infer_imgs = get_image_file_list("./test.jpg")

    
    with open(
            os.path.join("./save_re",
                         "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, info in enumerate(infer_imgs):
            if global_config.get("infer_mode", None) is False:
                data_line=info.decode("utf-8")
                substr=data_line.strip("\n").split("\t")
                img_path=os.path.join(data_dir,substr[0])
                data= {"img_path":img_path,"label": substring[1]}
            else:
                img_path = info
                data = {'img_path': img_path}

            save_img_path = os.path.join(
                "./save_re",
                os.path.splitext(os.path.basename(img_path))[0] + "_ser_re.jpg")

            result = ser_re_engine(data)

            result = result[0]
            fout.write(img_path + "\t" + json.dumps(
                result, ensure_ascii=False) + "\n")
            img_res = draw_re_results(img_path, result,font_path="./simfang.ttf")
            cv2.imwrite(save_img_path, img_res)
