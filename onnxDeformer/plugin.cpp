// Plugin Registration
#include <maya/MFnPlugin.h>
#include <maya/MStatus.h>

#include "include/onnxDeformer.h"

MStatus initializePlugin(MObject obj)
{
    MFnPlugin plugin(obj, "Rikki", "1.0", "Any");
    MStatus status = plugin.registerNode("inflateDeformerONNX", InflateDeformerONNX::s_id, InflateDeformerONNX::creator,
                                         InflateDeformerONNX::initialize, MPxNode::kDeformerNode);
    if (!status)
    {
        status.perror("registerNode");
        return status;
    }
    return status;
}

MStatus uninitializePlugin(MObject obj)
{
    MFnPlugin plugin(obj);
    MStatus status = plugin.deregisterNode(InflateDeformerONNX::s_id);
    if (!status)
    {
        status.perror("deregisterNode");
        return status;
    }
    return status;
}
