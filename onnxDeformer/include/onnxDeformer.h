#include <onnxruntime_cxx_api.h>

#include <maya/MPxDeformerNode.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MItGeometry.h>
#include <maya/MPoint.h>
#include <maya/MGlobal.h>

class InflateDeformerONNX : public MPxDeformerNode {
public:
    static void* creator() { return new InflateDeformerONNX(); }
    static MStatus initialize();
    static void initializeONNX();

    MStatus deform(MDataBlock& dataBlock, MItGeometry& iter, const MMatrix& localToWorldMatrix, unsigned int multiIndex) override;

    public:
    static MTypeId s_id;
    static MObject s_InflateAmount;

    static std::shared_ptr<Ort::Session> s_session;
    static bool s_isONNXInitialized;
    static Ort::Env* s_env;
};
