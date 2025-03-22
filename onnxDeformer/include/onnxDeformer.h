#include <onnxruntime_cxx_api.h>

#include <maya/MCallbackIdArray.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MPoint.h>
#include <maya/MPxDeformerNode.h>

class InflateDeformerONNX : public MPxDeformerNode
{
  public:
    static void *creator()
    {
        return new InflateDeformerONNX();
    }
    static MStatus initialize();
    void initializeONNX();
    void postConstructor();

    MStatus deform(MDataBlock &dataBlock, MItGeometry &iter, const MMatrix &localToWorldMatrix,
                   unsigned int multiIndex) override;

  public:
    static MTypeId s_id;
    static MObject s_inflate;
    static MObject s_onnxModelPath;

    static MCallbackId s_attributeChangedCallbackId;
    static std::shared_ptr<Ort::Session> s_session;
    static bool s_isONNXInitialized;
    static Ort::Env *s_env;
    static std::string s_loadedModelPath;

  protected:
    virtual ~InflateDeformerONNX();
};
