from NLP_classifier.Utils.utils import dic_to_file, file_to_dic, map_save_label


def convert_labels_to_id_and_save(labels, label_map_file):
    labels_id = {}
    i = 0
    for label in labels:
        if label not in labels_id:
            labels_id[label] = i
            i += 1
    dic_to_file(labels_id, label_map_file)
    return labels_id


def get_labels_id(labels, label_map_file, label_vec_file):
    labels_id = file_to_dic(label_map_file)
    map_save_label(labels, labels_id, label_vec_file)
