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
    std::shared_ptr<Ort::Session> m_session = nullptr;
    bool m_isONNXInitialized = false;
    Ort::Env *m_env = nullptr;
    std::string m_loadedModelPath = "";
    int m_inputFeatures = 7;

  protected:
    virtual ~InflateDeformerONNX();
};
