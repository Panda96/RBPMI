# -*- coding:utf-8 -*-
import xml.etree.ElementTree as eTree
import helper.detector_helper as helper

eTree.register_namespace("bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
eTree.register_namespace("bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")
eTree.register_namespace("dc", "http://www.omg.org/spec/DD/20100524/DC")
eTree.register_namespace("di", "http://www.omg.org/spec/DD/20100524/DI")


def get_element_id(ele_path, all_elements):
    for i, path in enumerate(all_elements):
        if path[3] == ele_path[3] and ele_path[0] == path[0] and ele_path[1] == path[1] and ele_path[2] == path[2]:
            return i
    print("element not found!")
    return -1


def create_bounds(rec):
    bounds = eTree.Element("{http://www.omg.org/spec/DD/20100524/DC}Bounds",
                           attrib={"x": str(rec[0]), "y": str(rec[1]), "width": str(rec[2]), "height": str(rec[3])})
    return bounds


def create_bpmn_shape(ele_id, ele_rec, other_attrib=None):
    shape_attrib = {"id": ele_id + "_shape", "bpmnElement": ele_id}
    if other_attrib is not None:
        for k, v in other_attrib.items():
            shape_attrib[k] = v
    bpmn_shape = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNShape", attrib=shape_attrib)
    bounds = create_bounds(ele_rec)
    bpmn_shape.append(bounds)
    return bpmn_shape


def create_bpmn_edge(flow_id, points):
    bpmn_edge = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNEdge",
                              attrib={"id": flow_id + "_edge", "bpmnElement": flow_id})
    for point in points:
        way_point = eTree.Element("{http://www.omg.org/spec/DD/20100524/DI}waypoint",
                                  attrib={"x": str(point[0]), "y": str(point[1])})
        bpmn_edge.append(way_point)
    return bpmn_edge


def create_event_definition(definition_type):
    event_definition = "{}EventDefinition".format(definition_type)
    return eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}" + event_definition)


def create_model(pools, all_elements, all_elements_type, all_seq_flows):
    # for pool_id, pool in enumerate(pools):
    #     print("pool_{}:".format(pool_id))
    #     lanes = pool["lanes"]
    #     elements = pool["elements"]
    #     for lane_id in range(len(lanes)):
    #         print("lane_{}:".format(lane_id))
    #         print(len(elements[lane_id]))
    definitions = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}definitions",
                                attrib={"id": "definitions", "name": "model"})
    collaboration = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}collaboration",
                                  attrib={"id": "collaboration", "isClosed": "true"})
    bpmn_diagram = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram",
                                 attrib={"name": "process_diagram", "id": "bpmn_diagram"})
    bpmn_plane = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane",
                               attrib={"id": "bpmn_plane", "bpmnElement": "collaboration"})
    bpmn_diagram.append(bpmn_plane)

    definitions.append(collaboration)
    # participant_list = []
    # process_list = []

    flows = []
    sub_procs_in_one_pool = []
    for pool_id, pool in enumerate(pools):
        participant_id = "participant_{}".format(pool_id)
        process_id = "process_{}".format(pool_id)

        # participant
        participant = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}participant",
                                    attrib={"id": participant_id, "processRef": process_id})
        collaboration.append(participant)

        pool_rect = pool["rect"]
        participant_shape = create_bpmn_shape(participant_id, pool_rect, {"isHorizontal": "true"})

        bpmn_plane.append(participant_shape)

        # participant ----------------

        # process
        process = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}process",
                                attrib={"id": process_id, "name": "process{}".format(pool_id), "processType": "None"})
        lane_set = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}laneSet",
                                 attrib={"id": process_id + "_lane_set"})

        process.append(lane_set)

        lanes = pool["lanes"]
        elements = pool["elements"]
        sub_procs = pool["sub_procs"]

        flows_id = set()

        for lane_id, lane in enumerate(lanes):
            # lane
            lane_ele_id = "pool_{}_lane_{}".format(pool_id, lane_id)
            lane_ele = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}lane",
                                     attrib={"id": lane_ele_id, "name": ""})

            lane_set.append(lane_ele)

            lane_ele_shape = create_bpmn_shape(lane_ele_id, lane)
            bpmn_plane.append(lane_ele_shape)
            # lane --------

            procs = sub_procs.get(lane_id, [])
            elements_in_lane = elements[lane_id]

            lane_sub_proc_nodes = []
            # subprocess flowNodeRef
            for proc_id, proc in enumerate(procs):
                e_id = get_element_id([pool_id, lane_id, proc_id, 1], all_elements)
                sub_proc_id = "subProcess_expanded_{}".format(e_id)
                sub_proc_node = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}subProcess",
                                              attrib={"id": sub_proc_id})
                process.append(sub_proc_node)
                lane_sub_proc_nodes.append(sub_proc_node)
                sub_procs_in_one_pool.append([proc, sub_proc_node])

                flow_node_ref = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}flowNodeRef")
                flow_node_ref.text = sub_proc_id
                lane_ele.append(flow_node_ref)

                sub_proc_shape = create_bpmn_shape(sub_proc_id, proc, {"isExpanded": "true"})
                bpmn_plane.append(sub_proc_shape)

            for ele_id, ele_rec in enumerate(elements_in_lane):
                e_id = get_element_id([pool_id, lane_id, ele_id, 0], all_elements)
                node_type = all_elements_type[e_id]

                node_id = "{}_{}".format(node_type, e_id)

                type_info = node_type.split("_")
                node_tag = type_info[0]

                node_shape = create_bpmn_shape(node_id, ele_rec)
                if "expanded" in type_info:
                    node_shape.attrib["isExpanded"] = "true"
                bpmn_plane.append(node_shape)

                is_boundary_event = False
                ele_node_id = ""
                if node_tag == "boundaryEvent" or node_tag == "intermediateCatchEvent":
                    for ele_id_e, ele_rec_e in enumerate(elements_in_lane):
                        if ele_id != ele_id_e and helper.is_overlap(ele_rec, ele_rec_e):
                            is_boundary_event = True
                            ele_e_id = get_element_id([pool_id, lane_id, ele_id_e, 0], all_elements)
                            ele_node_id = "{}_{}".format(all_elements_type[ele_e_id], ele_e_id)
                            break

                    if not is_boundary_event:
                        for proc_id, proc in enumerate(procs):
                            if helper.is_overlap(proc, ele_rec) and not helper.is_in(proc, ele_rec):
                                is_boundary_event = True
                                ele_node_id = lane_sub_proc_nodes[proc_id].attrib["id"]
                                break

                if is_boundary_event:
                    node_element = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}boundaryEvent",
                                                 attrib={"id": node_id})
                    node_element.attrib["attachedToRef"] = ele_node_id
                else:
                    node_element = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}" + node_tag,
                                                 attrib={"id": node_id})

                if node_tag.endswith("Event"):
                    if len(type_info) > 1:
                        if type_info[1] != "isInterrupting":
                            event_definition = create_event_definition(type_info[1])
                            node_element.append(event_definition)
                        if "isInterrupting" in type_info:
                            node_element.attrib["isInterrupting"] = "false"

                        if "cancelActivity" in type_info:
                            node_element.attrib["cancelActivity"] = "false"

                if node_tag == "subProcess":
                    if "triggeredByEvent" in type_info:
                        node_element.attrib["triggeredByEvent"] = "true"

                for flow_id, seq_flow in enumerate(all_seq_flows):
                    if seq_flow[0] == e_id:
                        incoming = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}incoming")
                        flow_element_id = "sequenceFlow_{}".format(flow_id)
                        incoming.text = flow_element_id
                        node_element.append(incoming)
                        flows_id.add(flow_id)
                    elif seq_flow[-1] == e_id:
                        outgoing = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}outgoing")
                        flow_element_id = "sequenceFlow_{}".format(flow_id)
                        outgoing.text = flow_element_id
                        node_element.append(outgoing)
                        flows_id.add(flow_id)

                node_in_sub_proc = False
                for proc_id, proc in enumerate(procs):
                    if helper.is_in(proc, ele_rec):
                        node_in_sub_proc = True
                        lane_sub_proc_nodes[proc_id].append(node_element)
                        break

                if not node_in_sub_proc:
                    flow_node_ref = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}flowNodeRef")
                    flow_node_ref.text = node_id
                    lane_ele.append(flow_node_ref)
                    process.append(node_element)

        flows_id = list(flows_id)
        for flow_id in flows_id:
            seq_flow_id = "sequenceFlow_{}".format(flow_id)
            seq_flow = all_seq_flows[flow_id]

            source_id = seq_flow[-1]
            target_id = seq_flow[0]
            source_ref = "{}_{}".format(all_elements_type[source_id], source_id)
            target_ref = "{}_{}".format(all_elements_type[target_id], target_id)

            seq_flow_element = eTree.Element("{http://www.omg.org/spec/BPMN/20100524/MODEL}sequenceFlow",
                                             attrib={"id": seq_flow_id, "sourceRef": source_ref,
                                                     "targetRef": target_ref})
            flow_in_sub_proc = False
            for proc, proc_node in sub_procs_in_one_pool:
                all_in = True
                for point in seq_flow[1]:
                    if not helper.point_is_in(proc, point):
                        all_in = False
                        break
                if all_in:
                    flow_in_sub_proc = True
                    proc_node.append(seq_flow_element)
                    break

            if not flow_in_sub_proc:
                process.append(seq_flow_element)

        flows.extend(flows_id)
        definitions.append(process)

    for flow_id in flows:
        seq_flow_id = "sequenceFlow_{}".format(flow_id)
        seq_flow = all_seq_flows[flow_id]
        seq_flow[1].reverse()
        bpmn_edge = create_bpmn_edge(seq_flow_id, seq_flow[1])
        bpmn_plane.append(bpmn_edge)
    definitions.append(bpmn_diagram)

    return definitions


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def export_xml(definitions, file_path):
    indent(definitions)
    tree = eTree.ElementTree(definitions)
    #
    # rawText = eTree.tostring(definitions)
    # document = dom.parseString(rawText)
    #
    # with open("output/{}.xml".format(file_name), "w", encoding="utf-8") as fh:
    #     document.writexml(fh, newl="\n", encoding="utf-8")
    # # tree = eTree.ElementTree(definitions)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)
