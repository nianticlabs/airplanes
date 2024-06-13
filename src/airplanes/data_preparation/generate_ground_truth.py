"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Please check the original file from PlaneRCNN repository:
https://github.com/NVlabs/planercnn/blob/master/data_prep/parse.py

python -m airplanes.data_preparation.generate_ground_truth
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData
from tqdm import tqdm

numPlanes = 200
numPlanesPerSegment = 2
planeAreaThreshold = 100
numIterations = 100
numIterationsPair = 1000
planeDiffThreshold = 0.05
fittingErrorThreshold = planeDiffThreshold
orthogonalThreshold = np.cos(np.deg2rad(60))
parallelThreshold = np.cos(np.deg2rad(30))


def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]
    return


def loadClassMap(scannet_folder):
    classMap = {}
    classLabelMap = {}
    with open(scannet_folder.parent / "scannetv2-labels.combined.tsv") as info_file:
        line_index = 0
        for line in info_file:
            if line_index > 0:
                line = line.split("\t")

                key = line[1].strip()
                classMap[key] = line[7].strip()
                classMap[key + "s"] = line[7].strip()
                classMap[key + "es"] = line[7].strip()
                classMap[key[:-1] + "ves"] = line[7].strip()

                if line[4].strip() != "":
                    nyuLabel = int(line[4].strip())
                else:
                    nyuLabel = -1
                    pass
                classLabelMap[key] = [nyuLabel, line_index - 1]
                classLabelMap[key + "s"] = [nyuLabel, line_index - 1]
                classLabelMap[key[:-1] + "ves"] = [nyuLabel, line_index - 1]
                pass
            line_index += 1
            continue
        pass
    return classMap, classLabelMap


def writePointCloudFace(filename, points, faces):
    with open(filename, "w") as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for point in points:
            for value in point[:3]:
                f.write(str(value) + " ")
                continue
            for value in point[3:]:
                f.write(str(int(value)) + " ")
                continue
            f.write("\n")
            continue
        for face in faces:
            f.write("3 " + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + "\n")
            continue
        f.close()
        pass
    return


def mergePlanes(
    points, planes, planePointIndices, planeSegments, segmentNeighbors, numPlanes, debug=False
):
    planeFittingErrors = []
    for plane, pointIndices in zip(planes, planePointIndices):
        XYZ = points[pointIndices]
        planeNorm = np.linalg.norm(plane)
        if planeNorm == 0:
            planeFittingErrors.append(fittingErrorThreshold * 2)
            continue
        diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / planeNorm
        planeFittingErrors.append(diff.mean())
        continue

    planeList = zip(planes, planePointIndices, planeSegments, planeFittingErrors)
    planeList = sorted(planeList, key=lambda x: x[3])

    while len(planeList) > 0:
        hasChange = False
        planeIndex = 0

        if debug:
            for index, planeInfo in enumerate(sorted(planeList, key=lambda x: -len(x[1]))):
                print(
                    index, planeInfo[0] / np.linalg.norm(planeInfo[0]), planeInfo[2], planeInfo[3]
                )
                continue
            pass

        while planeIndex < len(planeList):
            plane, pointIndices, segments, fittingError = planeList[planeIndex]
            if fittingError > fittingErrorThreshold:
                break
            neighborSegments = []
            for segment in segments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]
                    pass
                continue
            neighborSegments += list(segments)
            neighborSegments = set(neighborSegments)
            bestNeighborPlane = (fittingErrorThreshold, -1, None)
            for neighborPlaneIndex, neighborPlane in enumerate(planeList):
                if neighborPlaneIndex <= planeIndex:
                    continue
                if not bool(neighborSegments & neighborPlane[2]):
                    continue
                neighborPlaneNorm = np.linalg.norm(neighborPlane[0])
                if neighborPlaneNorm < 1e-4:
                    continue
                dotProduct = np.abs(
                    np.dot(neighborPlane[0], plane)
                    / np.maximum(neighborPlaneNorm * np.linalg.norm(plane), 1e-4)
                )
                if dotProduct < orthogonalThreshold:
                    continue
                newPointIndices = np.concatenate([neighborPlane[1], pointIndices], axis=0)
                XYZ = points[newPointIndices]
                if (
                    dotProduct > parallelThreshold
                    and len(neighborPlane[1]) > len(pointIndices) * 0.5
                ):
                    newPlane = fitPlane(XYZ)
                else:
                    newPlane = plane
                    pass
                diff = np.abs(np.matmul(XYZ, newPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(
                    newPlane
                )
                newFittingError = diff.mean()
                if debug:
                    print(
                        len(planeList),
                        planeIndex,
                        neighborPlaneIndex,
                        newFittingError,
                        plane / np.linalg.norm(plane),
                        neighborPlane[0] / np.linalg.norm(neighborPlane[0]),
                        dotProduct,
                        orthogonalThreshold,
                    )
                    pass
                if newFittingError < bestNeighborPlane[0]:
                    newPlaneInfo = [
                        newPlane,
                        newPointIndices,
                        segments.union(neighborPlane[2]),
                        newFittingError,
                    ]
                    bestNeighborPlane = (newFittingError, neighborPlaneIndex, newPlaneInfo)
                    pass
                continue
            if bestNeighborPlane[1] != -1:
                newPlaneList = (
                    planeList[:planeIndex]
                    + planeList[planeIndex + 1 : bestNeighborPlane[1]]
                    + planeList[bestNeighborPlane[1] + 1 :]
                )
                newFittingError, newPlaneIndex, newPlane = bestNeighborPlane
                for newPlaneIndex in range(len(newPlaneList)):
                    if (
                        (newPlaneIndex == 0 and newPlaneList[newPlaneIndex][3] > newFittingError)
                        or newPlaneIndex == len(newPlaneList) - 1
                        or (
                            newPlaneList[newPlaneIndex][3] < newFittingError
                            and newPlaneList[newPlaneIndex + 1][3] > newFittingError
                        )
                    ):
                        newPlaneList.insert(newPlaneIndex, newPlane)
                        break
                    continue
                if len(newPlaneList) == 0:
                    newPlaneList = [newPlane]
                    pass
                planeList = newPlaneList
                hasChange = True
            else:
                planeIndex += 1
                pass
            continue
        if not hasChange:
            break
        continue

    planeList = sorted(planeList, key=lambda x: -len(x[1]))

    minNumPlanes, maxNumPlanes = numPlanes
    if minNumPlanes == 1 and len(planeList) == 0:
        if debug:
            print("at least one plane")
            pass
    elif len(planeList) > maxNumPlanes:
        if debug:
            print("too many planes", len(planeList), maxNumPlanes)
            pass
        planeList = planeList[:maxNumPlanes] + [
            (np.zeros(3), planeInfo[1], planeInfo[2], fittingErrorThreshold)
            for planeInfo in planeList[maxNumPlanes:]
        ]
        pass

    groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments, groupedPlaneFittingErrors = zip(
        *planeList
    )
    return groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments


def process_scene(scene_id, scannet_folder, output_folder):
    filename = scannet_folder / scene_id / (scene_id + ".aggregation.json")
    data = json.load(open(filename, "r"))
    aggregation = np.array(data["segGroups"])

    high_res = False

    if high_res:
        filename = scannet_folder / scene_id / (scene_id + "_vh_clean.labels.ply")
    else:
        filename = scannet_folder / scene_id / (scene_id + "_vh_clean_2.labels.ply")

    plydata = PlyData.read(filename)
    vertices = plydata["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    faces = np.array(plydata["face"]["vertex_indices"])

    if high_res:
        filename = scannet_folder / scene_id / (scene_id + "_vh_clean.segs.json")
    else:
        filename = scannet_folder / scene_id / (scene_id + "_vh_clean_2.0.010000.segs.json")

    data = json.load(open(filename, "r"))
    segmentation = np.array(data["segIndices"])

    groupSegments = []
    groupLabels = []
    for segmentIndex in range(len(aggregation)):
        groupSegments.append(aggregation[segmentIndex]["segments"])
        groupLabels.append(aggregation[segmentIndex]["label"])

    segmentation = segmentation.astype(np.int32)

    uniqueSegments = np.unique(segmentation).tolist()
    numSegments = 0
    for segments in groupSegments:
        for segmentIndex in segments:
            if segmentIndex in uniqueSegments:
                uniqueSegments.remove(segmentIndex)

        numSegments += len(segments)

    for segment in uniqueSegments:
        groupSegments.append(
            [
                segment,
            ]
        )
        groupLabels.append("unannotated")
        continue

    numPoints = segmentation.shape[0]
    numPlanes = 1000

    segmentEdges = []
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = segmentation[face[0]]
        segment_2 = segmentation[face[1]]
        segment_3 = segmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            if segment_1 != segment_2 and segment_1 != -1 and segment_2 != -1:
                segmentEdges.append((min(segment_1, segment_2), max(segment_1, segment_2)))

            if segment_1 != segment_3 and segment_1 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_1, segment_3), max(segment_1, segment_3)))

            if segment_2 != segment_3 and segment_2 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_2, segment_3), max(segment_2, segment_3)))

    segmentEdges = list(set(segmentEdges))

    labelNumPlanes = {
        "wall": [1, 3],
        "floor": [1, 1],
        "cabinet": [0, 5],
        "bed": [0, 5],
        "chair": [0, 5],
        "sofa": [0, 10],
        "table": [0, 5],
        "door": [1, 2],
        "window": [0, 2],
        "bookshelf": [0, 5],
        "picture": [1, 1],
        "counter": [0, 10],
        "blinds": [0, 0],
        "desk": [0, 10],
        "shelf": [0, 5],
        "shelves": [0, 5],
        "curtain": [0, 0],
        "dresser": [0, 5],
        "pillow": [0, 0],
        "mirror": [0, 0],
        "entrance": [1, 1],
        "floor mat": [1, 1],
        "clothes": [0, 0],
        "ceiling": [0, 5],
        "book": [0, 1],
        "books": [0, 1],
        "refridgerator": [0, 5],
        "television": [1, 1],
        "paper": [0, 1],
        "towel": [0, 1],
        "shower curtain": [0, 1],
        "box": [0, 5],
        "whiteboard": [1, 5],
        "person": [0, 0],
        "night stand": [1, 5],
        "toilet": [0, 5],
        "sink": [0, 5],
        "lamp": [0, 1],
        "bathtub": [0, 5],
        "bag": [0, 1],
        "otherprop": [0, 5],
        "otherstructure": [0, 5],
        "otherfurniture": [0, 5],
        "unannotated": [0, 5],
        "": [0, 0],
    }
    nonPlanarGroupLabels = ["bicycle", "bottle", "water bottle"]
    nonPlanarGroupLabels = {label: True for label in nonPlanarGroupLabels}

    classMap, classLabelMap = loadClassMap(scannet_folder=scannet_folder)
    classMap["unannotated"] = "unannotated"
    classLabelMap["unannotated"] = [max([index for index, label in classLabelMap.values()]) + 1, 41]
    allXYZ = points.reshape(-1, 3)

    segmentNeighbors = {}
    for segmentEdge in segmentEdges:
        if segmentEdge[0] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[0]] = []

        segmentNeighbors[segmentEdge[0]].append(segmentEdge[1])

        if segmentEdge[1] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[1]] = []

        segmentNeighbors[segmentEdge[1]].append(segmentEdge[0])

    planeGroups = []

    debugIndex = -1

    for groupIndex, group in enumerate(groupSegments):
        if debugIndex != -1 and groupIndex != debugIndex:
            continue
        if groupLabels[groupIndex] in nonPlanarGroupLabels:
            groupLabel = groupLabels[groupIndex]
            minNumPlanes, maxNumPlanes = 0, 0
        elif groupLabels[groupIndex] in classMap:
            groupLabel = classMap[groupLabels[groupIndex]]
            minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]
        else:
            minNumPlanes, maxNumPlanes = 0, 0
            groupLabel = ""

        if maxNumPlanes == 0:
            pointMasks = []
            for segmentIndex in group:
                pointMasks.append(segmentation == segmentIndex)

            pointIndices = np.any(np.stack(pointMasks, 0), 0).nonzero()[0]
            groupPlanes = [[np.zeros(3), pointIndices, []]]
            planeGroups.append(groupPlanes)
            continue

        groupPlanes = []
        groupPlanePointIndices = []
        groupPlaneSegments = []
        for segmentIndex in group:
            segmentMask = segmentation == segmentIndex
            allSegmentIndices = segmentMask.nonzero()[0]
            segmentIndices = allSegmentIndices.copy()

            XYZ = allXYZ[segmentMask.reshape(-1)]
            numPoints = XYZ.shape[0]

            for c in range(2):
                if c == 0:
                    ## First try to fit one plane
                    plane = fitPlane(XYZ)
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(
                        plane
                    )
                    if diff.mean() < fittingErrorThreshold:
                        groupPlanes.append(plane)
                        groupPlanePointIndices.append(segmentIndices)
                        groupPlaneSegments.append(set([segmentIndex]))
                        break
                else:
                    ## Run ransac
                    segmentPlanes = []
                    segmentPlanePointIndices = []

                    for planeIndex in range(numPlanesPerSegment):
                        if len(XYZ) < planeAreaThreshold:
                            continue
                        bestPlaneInfo = [None, 0, None]
                        for iteration in range(min(XYZ.shape[0], numIterations)):
                            sampledPoints = XYZ[
                                np.random.choice(np.arange(XYZ.shape[0]), size=(3), replace=False)
                            ]
                            try:
                                plane = fitPlane(sampledPoints)

                            except:
                                continue
                            diff = np.abs(
                                np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])
                            ) / np.linalg.norm(plane)
                            inlierMask = diff < planeDiffThreshold
                            numInliers = inlierMask.sum()
                            if numInliers > bestPlaneInfo[1]:
                                bestPlaneInfo = [plane, numInliers, inlierMask]

                        if bestPlaneInfo[1] < planeAreaThreshold:
                            break

                        pointIndices = segmentIndices[bestPlaneInfo[2]]
                        bestPlane = fitPlane(XYZ[bestPlaneInfo[2]])

                        segmentPlanes.append(bestPlane)
                        segmentPlanePointIndices.append(pointIndices)

                        outlierMask = np.logical_not(bestPlaneInfo[2])
                        segmentIndices = segmentIndices[outlierMask]
                        XYZ = XYZ[outlierMask]

                    if (
                        sum([len(indices) for indices in segmentPlanePointIndices])
                        < numPoints * 0.5
                    ):
                        groupPlanes.append(np.zeros(3))
                        groupPlanePointIndices.append(allSegmentIndices)
                        groupPlaneSegments.append(set([segmentIndex]))
                    else:
                        if len(segmentIndices) > 0:
                            ## Add remaining non-planar regions
                            segmentPlanes.append(np.zeros(3))
                            segmentPlanePointIndices.append(segmentIndices)

                        groupPlanes += segmentPlanes
                        groupPlanePointIndices += segmentPlanePointIndices

                        for _ in range(len(segmentPlanes)):
                            groupPlaneSegments.append(set([segmentIndex]))

        numRealPlanes = len([plane for plane in groupPlanes if np.linalg.norm(plane) > 1e-4])
        if minNumPlanes == 1 and numRealPlanes == 0:
            ## Some instances always contain at least one planes (e.g, the floor)
            maxArea = (planeAreaThreshold, -1)
            for index, indices in enumerate(groupPlanePointIndices):
                if len(indices) > maxArea[0]:
                    maxArea = (len(indices), index)

            maxArea, planeIndex = maxArea
            if planeIndex >= 0:
                groupPlanes[planeIndex] = fitPlane(allXYZ[groupPlanePointIndices[planeIndex]])
                numRealPlanes = 1

        if minNumPlanes == 1 and maxNumPlanes == 1 and numRealPlanes > 1:
            ## Some instances always contain at most one planes (e.g, the floor)

            pointIndices = np.concatenate(
                [indices for plane, indices in zip(groupPlanes, groupPlanePointIndices)], axis=0
            )
            XYZ = allXYZ[pointIndices]
            plane = fitPlane(XYZ)
            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)

            if groupLabel == "floor":
                ## Relax the constraint for the floor due to the misalignment issue in ScanNet
                fittingErrorScale = 3
            else:
                fittingErrorScale = 1

            if diff.mean() < fittingErrorThreshold * fittingErrorScale:
                groupPlanes = [plane]
                groupPlanePointIndices = [pointIndices]
                planeSegments = []
                for segments in groupPlaneSegments:
                    planeSegments += list(segments)

                groupPlaneSegments = [set(planeSegments)]
                numRealPlanes = 1

        if numRealPlanes > 1:
            groupPlanes, groupPlanePointIndices, groupPlaneSegments = mergePlanes(
                points,
                groupPlanes,
                groupPlanePointIndices,
                groupPlaneSegments,
                segmentNeighbors,
                numPlanes=(minNumPlanes, maxNumPlanes),
                debug=debugIndex != -1,
            )

        groupNeighbors = []
        for planeIndex, planeSegments in enumerate(groupPlaneSegments):
            neighborSegments = []
            for segment in planeSegments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]

            neighborSegments += list(planeSegments)
            neighborSegments = set(neighborSegments)
            neighborPlaneIndices = []
            for neighborPlaneIndex, neighborPlaneSegments in enumerate(groupPlaneSegments):
                if neighborPlaneIndex == planeIndex:
                    continue
                if bool(neighborSegments & neighborPlaneSegments):
                    plane = groupPlanes[planeIndex]
                    neighborPlane = groupPlanes[neighborPlaneIndex]
                    if np.linalg.norm(plane) * np.linalg.norm(neighborPlane) < 1e-4:
                        continue
                    dotProduct = np.abs(
                        np.dot(plane, neighborPlane)
                        / np.maximum(np.linalg.norm(plane) * np.linalg.norm(neighborPlane), 1e-4)
                    )
                    if dotProduct < orthogonalThreshold:
                        neighborPlaneIndices.append(neighborPlaneIndex)

            groupNeighbors.append(neighborPlaneIndices)

        groupPlanes = list(zip(groupPlanes, groupPlanePointIndices, groupNeighbors))
        planeGroups.append(groupPlanes)

    numPlanes = sum([len(group) for group in planeGroups])
    segmentationColor = (np.arange(numPlanes + 1) + 1) * 100
    colorMap = np.stack(
        [
            segmentationColor / (256 * 256),
            segmentationColor / 256 % 256,
            segmentationColor % 256,
        ],
        axis=1,
    )
    colorMap[-1] = 0

    annotationFolder = output_folder / scene_id
    annotationFolder.mkdir(exist_ok=True, parents=True)

    planes = []
    planePointIndices = []
    planeInfo = []
    structureIndex = 0
    for index, group in enumerate(planeGroups):
        groupPlanes, groupPlanePointIndices, groupNeighbors = zip(*group)

        diag = np.diag(np.ones(len(groupNeighbors)))
        adjacencyMatrix = diag.copy()
        for groupIndex, neighbors in enumerate(groupNeighbors):
            for neighbor in neighbors:
                adjacencyMatrix[groupIndex][neighbor] = 1
        if groupLabels[index] in classLabelMap:
            label = classLabelMap[groupLabels[index]]
        else:
            print("label not valid", groupLabels[index])
            exit(1)
            label = -1
        groupInfo = [[(index, label[0], label[1])] for _ in range(len(groupPlanes))]
        groupPlaneIndices = (adjacencyMatrix.sum(-1) >= 2).nonzero()[0]
        usedMask = {}
        for groupPlaneIndex in groupPlaneIndices:
            if groupPlaneIndex in usedMask:
                continue
            groupStructure = adjacencyMatrix[groupPlaneIndex].copy()
            for neighbor in groupStructure.nonzero()[0]:
                if np.any(adjacencyMatrix[neighbor] < groupStructure):
                    groupStructure[neighbor] = 0

            groupStructure = groupStructure.nonzero()[0]

            if len(groupStructure) < 2:
                print("invalid structure")
                print(groupPlaneIndex, groupPlaneIndices)
                print(groupNeighbors)
                print(groupPlaneIndex)
                print(adjacencyMatrix.sum(-1) >= 2)
                print((adjacencyMatrix.sum(-1) >= 2).nonzero()[0])
                print(adjacencyMatrix[groupPlaneIndex])
                print(adjacencyMatrix)
                print(groupStructure)
                exit(1)

            if len(groupStructure) >= 4:
                print("complex structure")
                print("group index", index)
                print(adjacencyMatrix)
                print(groupStructure)
                groupStructure = groupStructure[:3]

            if len(groupStructure) in [2, 3]:
                for planeIndex in groupStructure:
                    groupInfo[planeIndex].append((structureIndex, len(groupStructure)))
                structureIndex += 1

            for planeIndex in groupStructure:
                usedMask[planeIndex] = True

        planes += groupPlanes
        planePointIndices += groupPlanePointIndices
        planeInfo += groupInfo

    planeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
    for planeIndex, planePoints in enumerate(planePointIndices):
        planeSegmentation[planePoints] = planeIndex

    planes = np.array(planes)
    planesD = 1.0 / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-4)
    planes *= pow(planesD, 2)

    removeIndices = []
    edge_verts = set()
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = planeSegmentation[face[0]]
        segment_2 = planeSegmentation[face[1]]
        segment_3 = planeSegmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            removeIndices.append(faceIndex)
            edge_verts.add(face[0])
            edge_verts.add(face[1])
            edge_verts.add(face[2])

    edge_verts = np.array(list(edge_verts), dtype="int32").tolist()
    colors_rgb = colorMap[planeSegmentation].astype("uint8")
    colors = np.ones((colors_rgb.shape[0], 4), dtype="uint8") * 255
    colors[:, :3] = colors_rgb
    colors[edge_verts, 3] = 0

    mesh = trimesh.Trimesh(points.tolist(), faces.tolist(), vertex_colors=colors)
    result = trimesh.exchange.ply.export_ply(mesh, encoding="ascii")
    with open(annotationFolder / "mesh_with_planes.ply", "wb+") as f:
        f.write(result)

    np.save(annotationFolder / f"{scene_id}_planes.npy", planes)
    return len(planes)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scannet", type=Path, required=True, help="Path to ScanNetv2 scans")
    parser.add_argument(
        "--output", type=Path, required=True, help="Where do you want to save outcomes?"
    )
    args = parser.parse_args()

    np.random.seed(10)
    random.seed(10)

    scene_ids = [scene.name for scene in args.scannet.iterdir()]
    print(f"found {len(scene_ids)} scenes")
    scene_ids = sorted(scene_ids)

    for index, scene_id in enumerate(tqdm(scene_ids)):
        if scene_id[:5] != "scene":
            continue
        num_planes = process_scene(scene_id, scannet_folder=args.scannet, output_folder=args.output)
