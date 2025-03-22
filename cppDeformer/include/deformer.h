#pragma once

#include <maya/MFloatVectorArray.h>
#include <maya/MFnMesh.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MPointArray.h>
#include <maya/MPxDeformerNode.h>

class InflateDeformer : public MPxDeformerNode
{
  public:
    InflateDeformer() = default;
    ~InflateDeformer() = default;

    static void *creator()
    {
        return new InflateDeformer();
    };
    static MStatus initialize();

    MStatus deform(MDataBlock &dataBlock, MItGeometry &iter, const MMatrix &localToWorldMatrix,
                   unsigned int multiIndex) override;

  public:
    static MTypeId s_id;
    static MObject s_InflateAmount;
};
