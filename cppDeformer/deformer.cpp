#include "include/deformer.h"

MTypeId InflateDeformer::s_id(0x00123456);
MObject InflateDeformer::s_InflateAmount;

MStatus InflateDeformer::initialize() {
    MFnNumericAttribute nAttr;
    s_InflateAmount = nAttr.create("inflateAmount", "ia", MFnNumericData::kFloat, 0.0);
    nAttr.setKeyable(true);
    addAttribute(s_InflateAmount);
    attributeAffects(s_InflateAmount, outputGeom);
    return MS::kSuccess;
}

MStatus InflateDeformer::deform(MDataBlock& dataBlock, MItGeometry& iter, const MMatrix& localToWorldMatrix, unsigned int multiIndex) {
    MDataHandle envHandle = dataBlock.inputValue(MPxDeformerNode::envelope);
    float env = envHandle.asFloat();
    MDataHandle inflateHandle = dataBlock.inputValue(s_InflateAmount);
    float inflateAmount = inflateHandle.asFloat();

    if (env == 0.0f || inflateAmount == 0.0f) {
        return MS::kSuccess;
    }

    MArrayDataHandle inputArray = dataBlock.inputArrayValue(input);
    inputArray.jumpToElement(multiIndex);
    MObject meshObj = inputArray.inputValue().child(inputGeom).asMesh();
    MFnMesh meshFn(meshObj);
    MFloatVectorArray normals;
    meshFn.getVertexNormals(true, normals);

    for (; !iter.isDone(); iter.next()) {
        int index = iter.index();
        MPoint pt = iter.position();
        MVector normal = normals[index];
        pt += normal * inflateAmount * env;
        iter.setPosition(pt);
    }

    return MS::kSuccess;
}
