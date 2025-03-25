#include <maya/MFnEnumAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MMessage.h>
#include <maya/MNodeMessage.h>
#include <maya/MPointArray.h>
#include <maya/MVectorArray.h>

#include "include/onnxDeformer.h"

// Initialize ONNX Runtime session
MTypeId InflateDeformerONNX::s_id(0x00123457);
MCallbackId InflateDeformerONNX::s_attributeChangedCallbackId = 0;
MObject InflateDeformerONNX::s_inflate;
MObject InflateDeformerONNX::s_onnxModelPath;

static void onAttributeChanged(MNodeMessage::AttributeMessage msg, MPlug &plug, MPlug &otherPlug, void *clientData)
{
    if (msg & MNodeMessage::kAttributeSet)
    {
        InflateDeformerONNX *deformer = static_cast<InflateDeformerONNX *>(clientData);

        if (plug == deformer->s_onnxModelPath)
        {
            deformer->initializeONNX();
        }
    }
}

InflateDeformerONNX::~InflateDeformerONNX()
{
    if (s_attributeChangedCallbackId != 0)
    {
        MMessage::removeCallback(s_attributeChangedCallbackId);
    }
}

MStatus InflateDeformerONNX::initialize()
{
    MFnNumericAttribute nAttr;
    s_inflate = nAttr.create("inflate", "i", MFnNumericData::kFloat, 0.f);
    nAttr.setKeyable(true);
    addAttribute(s_inflate);
    attributeAffects(s_inflate, outputGeom);

    MFnTypedAttribute tAttr;
    s_onnxModelPath = tAttr.create("onnxModelPath", "omp", MFnData::kString);
    tAttr.setStorable(true);
    tAttr.setKeyable(false);
    tAttr.setWritable(true);
    tAttr.setUsedAsFilename(true);
    addAttribute(s_onnxModelPath);
    attributeAffects(s_onnxModelPath, outputGeom);

    return MS::kSuccess;
}

// Load the ONNX model and initialize the InferenceSession
void InflateDeformerONNX::initializeONNX()
{
    if (!m_env)
    {
        m_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXMayaDeformer");
    }

    MPlug modelPathPlug(thisMObject(), s_onnxModelPath);
    const std::string modelPath = modelPathPlug.asString().asChar();
    const std::wstring wmodelPath(modelPath.begin(), modelPath.end());

    if (m_isONNXInitialized && m_loadedModelPath == modelPath)
    {
        return;
    }

    // If a session already exists, release it before loading a new one
    if (m_session)
    {
        m_session.reset();
    }

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Load the ONNX model
    try
    {
        m_session = std::make_shared<Ort::Session>(*m_env, wmodelPath.c_str(), sessionOptions);

        if (m_session->GetInputCount() == 0)
        {
            MGlobal::displayError("Failed to load ONNX model. No inputs found.");
            m_session.reset(); // Ensure no invalid session remains
            return;
        }

        // Get the input tensor's shape dynamically
        const Ort::TypeInfo& input_type_info = m_session->GetInputTypeInfo(0);  // Assuming the first input is the relevant one
        std::vector<int64_t> input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

        // The last dimension is the number of features
        int inputFeatures = static_cast<int>(input_shape.back());  // Get the feature size from the input shape

        // If the number of features is valid, proceed
        if (inputFeatures <= 0)
        {
            MGlobal::displayError("Invalid number of input features in ONNX model.");
            m_session.reset(); // Ensure no invalid session remains
            return;
        }
        else if (inputFeatures != m_inputFeatures)
        {
            MGlobal::displayError(("Input features to not match expecting " + std::to_string(m_inputFeatures) + " got " + std::to_string(inputFeatures)).c_str());
            m_session.reset(); // Ensure no invalid session remains
            return;
        }

        // Update loaded model path
        m_loadedModelPath = modelPath;
        m_isONNXInitialized = true;
    }
    catch (const std::exception &e)
    {
        MGlobal::displayError(MString("Failed to load ONNX model: ") + (modelPath + ". " + e.what()).c_str());
        m_session.reset(); // Ensure no invalid session remains
    }
}

void InflateDeformerONNX::postConstructor()
{
    MStatus status;

    MObject thisNode = thisMObject();
    s_attributeChangedCallbackId = MNodeMessage::addAttributeChangedCallback(thisNode, onAttributeChanged, this);
}

// Deform function with ONNX inference for inflation prediction
MStatus InflateDeformerONNX::deform(MDataBlock &dataBlock, MItGeometry &iter, const MMatrix &localToWorldMatrix,
                                    unsigned int multiIndex)
{
    // If the ONNX model is not initialized, exit early
    if (!m_isONNXInitialized)
    {
        return MS::kFailure;
    }

    MDataHandle envH = dataBlock.inputValue(envelope);
    const float envWeight = envH.asFloat();
    // Skip deformation if envelope weight is zero
    if (envWeight == 0.0f)
    {
        return MS::kSuccess;
    }

    MDataHandle inflateH = dataBlock.inputValue(s_inflate);
    const float inflateWeight = inflateH.asFloat();

    MPointArray positions;
    iter.allPositions(positions);
    auto num_vertices = positions.length();
    MVectorArray normals;
    normals.setLength(num_vertices);
    iter.reset();
    for (; iter.isDone(); iter.next())
    {
        normals.append(iter.normal());
    }

    // Pre-allocate memory for input data (7 floats per vertex: posx, posy, posz, normx, normy, normz, inflate)
    std::vector<float> input_data(num_vertices * m_inputFeatures);

    // Populate input data for ONNX model with normalization
    for (unsigned int i = 0; i < num_vertices; ++i)
    {
        input_data[i * m_inputFeatures + 0] = static_cast<float>(positions[i].x);
        input_data[i * m_inputFeatures + 1] = static_cast<float>(positions[i].y);
        input_data[i * m_inputFeatures + 2] = static_cast<float>(positions[i].z);
        input_data[i * m_inputFeatures + 3] = static_cast<float>(normals[i].x);
        input_data[i * m_inputFeatures + 4] = static_cast<float>(normals[i].y);
        input_data[i * m_inputFeatures + 5] = static_cast<float>(normals[i].z);
        input_data[i * m_inputFeatures + 6] = inflateWeight;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // Prepare input tensor for ONNX
    std::vector<int64_t> input_shape = {static_cast<int64_t>(num_vertices), m_inputFeatures};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                              input_shape.data(), input_shape.size());

    // Output tensor to hold deltas (3 values per vertex: delta_x, delta_y, delta_z)
    const int output_features = 3;
    std::vector<float> output_data(num_vertices * output_features);
    std::vector<int64_t> output_shape = {static_cast<int64_t>(num_vertices), output_features};

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_data.data(), output_data.size(),
                                                               output_shape.data(), output_shape.size());

    // Run inference
    try
    {
        Ort::RunOptions run_options;
        const char *input_names[] = {"input_name"};
        const char *output_names[] = {"output_name"};

        m_session->Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    }
    catch (const Ort::Exception &e)
    {
        MGlobal::displayError(MString("ONNX Runtime Error: ") + e.what());
        return MS::kFailure;
    }

    iter.reset();

    // Apply deformation using the predicted deltas with smoothing
    for (unsigned int i = 0; i < num_vertices; ++i)
    {
        float delta_x = output_data[i * output_features + 0];
        float delta_y = output_data[i * output_features + 1];
        float delta_z = output_data[i * output_features + 2];

        MPoint pt = iter.position();
        pt.x += delta_x * envWeight;
        pt.y += delta_y * envWeight;
        pt.z += delta_z * envWeight;

        iter.setPosition(pt);
        iter.next();
    }

    return MS::kSuccess;
}
