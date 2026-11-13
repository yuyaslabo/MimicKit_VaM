
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class EnvironmentBoxSender : MVRScript
{
    private UdpClient udpClient;
    private string ipAddress = "127.0.0.1";
    // VaM UI: destination IP and both UDP ports.
    private JSONStorableFloat ipOctet1JSON;
    private JSONStorableFloat ipOctet2JSON;
    private JSONStorableFloat ipOctet3JSON;
    private JSONStorableFloat ipOctet4JSON;
    private JSONStorableFloat posePortJSON;
    private JSONStorableFloat shapePortJSON;
    private int port = 9001;
    // Separate shape port. Original packet formats remain unchanged.
    private int shapePort = 9002;
    private const float ShapeMagic = 424242f;
    private const float ShapeVersion = 1f;
    private HashSet<string> shapeSentUids = new HashSet<string>();

    private float sendInterval = 0.5f;
    private float timeSinceLastSend = 0f;

    private float logInterval = 1.0f;
    private float timeSinceLastLog = 0f;
	
	private Dictionary<string, int> atomIds = new Dictionary<string, int>();
	private int nextAtomId = 1;

    // Exclude VaM UI/navigation Atoms and non-physical scene helpers.
    // No allowlist: newly added furniture Atoms are eligible automatically.
    private bool IsEligibleAtom(Atom atom)
    {
        if (atom == null) return false;
        string nameLower = atom.name.ToLowerInvariant();
        string typeLower = atom.type.ToLowerInvariant();
        if (nameLower == "playernavigationpanel" || nameLower == "slate") return false;
        if (typeLower == "control" || typeLower == "person") return false;
        if (nameLower.Contains("camera") || nameLower.Contains("control") ||
            nameLower.Contains("rig") || nameLower.Contains("audio") ||
            nameLower.Contains("dancer") || nameLower.Contains("hand") ||
            nameLower.Contains("hmd") || nameLower.Contains("light") ||
            nameLower.Contains("showroom") || nameLower.Contains("stage"))
            return false;
        return true;
    }

    // Ignore VaM manipulation handles, not part of the physical bench.
    private bool IsPhysicalCollider(Collider col)
    {
        return col != null && !col.isTrigger && col.enabled &&
               col.gameObject.activeInHierarchy &&
               !string.Equals(col.gameObject.name, "control", StringComparison.OrdinalIgnoreCase);
    }

    // Non-convex mesh bounds often enclose the entire decorative model.
    // Prefer separately authored convex collision proxies when available.
    private bool IsTransferCollider(Collider col)
    {
        if (!IsPhysicalCollider(col)) return false;
        MeshCollider mesh = col as MeshCollider;
        return mesh == null || mesh.convex;
    }

    // Collider diagnostics: log each Atom only once.
    private bool colliderDiagnosticsEnabled = true;
    private HashSet<int> diagnosedAtomIds = new HashSet<int>();

	private int GetAtomId(Atom atom)
	{
		string uid = atom.uid;

		// FNV-1a 32bit
		uint hash = 2166136261u;

		for (int i = 0; i < uid.Length; i++)
		{
			hash ^= (uint)uid[i];
			hash *= 16777619u;
		}

		// float32で整数を正確に保持できる24bitに制限
		return (int)(hash & 0x00FFFFFFu);
	}

    public override void Init()
    {
        udpClient = new UdpClient();

        // Same four-octet slider UI pattern as VRhand_Mimickit2.cs.
        // Loopback default for single-PC setups.
        ipOctet1JSON = new JSONStorableFloat("IP Octet 1", 127f, 0f, 255f, true, true);
        ipOctet1JSON.setCallbackFunction = delegate (float val) { UpdateIPAddress(); };
        UIDynamicSlider slider1 = CreateSlider(ipOctet1JSON);
        if (slider1 != null) { slider1.label = "IP Address [1]"; slider1.valueFormat = "F0"; slider1.quickButtonsEnabled = true; }
        RegisterFloat(ipOctet1JSON);

        ipOctet2JSON = new JSONStorableFloat("IP Octet 2", 0f, 0f, 255f, true, true);
        ipOctet2JSON.setCallbackFunction = delegate (float val) { UpdateIPAddress(); };
        UIDynamicSlider slider2 = CreateSlider(ipOctet2JSON);
        if (slider2 != null) { slider2.label = "IP Address [2]"; slider2.valueFormat = "F0"; slider2.quickButtonsEnabled = true; }
        RegisterFloat(ipOctet2JSON);

        ipOctet3JSON = new JSONStorableFloat("IP Octet 3", 0f, 0f, 255f, true, true);
        ipOctet3JSON.setCallbackFunction = delegate (float val) { UpdateIPAddress(); };
        UIDynamicSlider slider3 = CreateSlider(ipOctet3JSON);
        if (slider3 != null) { slider3.label = "IP Address [3]"; slider3.valueFormat = "F0"; slider3.quickButtonsEnabled = true; }
        RegisterFloat(ipOctet3JSON);

        ipOctet4JSON = new JSONStorableFloat("IP Octet 4", 1f, 0f, 255f, true, true);
        ipOctet4JSON.setCallbackFunction = delegate (float val) { UpdateIPAddress(); };
        UIDynamicSlider slider4 = CreateSlider(ipOctet4JSON);
        if (slider4 != null) { slider4.label = "IP Address [4]"; slider4.valueFormat = "F0"; slider4.quickButtonsEnabled = true; }
        RegisterFloat(ipOctet4JSON);

        // Configurable UDP destination ports (same slider pattern as VRhand_Mimickit2.cs).
        posePortJSON = new JSONStorableFloat("Pose Port", 9001f, 1024f, 65535f, true, true);
        posePortJSON.setCallbackFunction = delegate (float val) { UpdatePorts(); };
        UIDynamicSlider posePortSlider = CreateSlider(posePortJSON);
        if (posePortSlider != null) { posePortSlider.label = "Pose UDP Port"; posePortSlider.valueFormat = "F0"; posePortSlider.quickButtonsEnabled = true; }
        RegisterFloat(posePortJSON);

        shapePortJSON = new JSONStorableFloat("Shape Port", 9002f, 1024f, 65535f, true, true);
        shapePortJSON.setCallbackFunction = delegate (float val) { UpdatePorts(); };
        UIDynamicSlider shapePortSlider = CreateSlider(shapePortJSON);
        if (shapePortSlider != null) { shapePortSlider.label = "Shape UDP Port"; shapePortSlider.valueFormat = "F0"; shapePortSlider.quickButtonsEnabled = true; }
        RegisterFloat(shapePortJSON);

        UpdatePorts();
        UpdateIPAddress();
        SuperController.LogMessage(
            "[EnvSender] Local OBB Mode Initialized on port " + port);
    }

    private void UpdateIPAddress()
    {
        if (ipOctet1JSON == null || ipOctet2JSON == null ||
            ipOctet3JSON == null || ipOctet4JSON == null) return;

        string updatedIP = string.Format("{0}.{1}.{2}.{3}",
            Mathf.RoundToInt(ipOctet1JSON.val),
            Mathf.RoundToInt(ipOctet2JSON.val),
            Mathf.RoundToInt(ipOctet3JSON.val),
            Mathf.RoundToInt(ipOctet4JSON.val));
        try
        {
            IPAddress.Parse(updatedIP);
            ipAddress = updatedIP;
            SuperController.LogMessage("[EnvSender] UDP Target Updated: " + ipAddress);
        }
        catch (Exception e)
        {
            SuperController.LogError("[EnvSender] Invalid IP Address: " + e.Message);
        }
    }

    private void UpdatePorts()
    {
        if (posePortJSON == null || shapePortJSON == null) return;
        port = Mathf.RoundToInt(posePortJSON.val);
        shapePort = Mathf.RoundToInt(shapePortJSON.val);
        SuperController.LogMessage(string.Format(
            "[EnvSender] UDP Ports Updated: pose={0}, shape={1}", port, shapePort));
    }

    // Colliderのローカル形状を取得
    private bool GetColliderLocalBounds(
        Collider col,
        out Vector3 center,
        out Vector3 size)
    {
        center = Vector3.zero;
        size = Vector3.zero;

        BoxCollider box = col as BoxCollider;
        if (box != null)
        {
            center = box.center;
            size = box.size;
            return true;
        }

        SphereCollider sphere = col as SphereCollider;
        if (sphere != null)
        {
            center = sphere.center;
            float d = sphere.radius * 2f;
            size = new Vector3(d, d, d);
            return true;
        }

        CapsuleCollider capsule = col as CapsuleCollider;
        if (capsule != null)
        {
            center = capsule.center;

            float d = capsule.radius * 2f;
            float h = Mathf.Max(capsule.height, d);

            size = new Vector3(d, d, d);

            if (capsule.direction == 0) size.x = h;
            if (capsule.direction == 1) size.y = h;
            if (capsule.direction == 2) size.z = h;

            return true;
        }

        MeshCollider meshCol = col as MeshCollider;
        if (meshCol != null && meshCol.sharedMesh != null)
        {
            Bounds b = meshCol.sharedMesh.bounds;
            center = b.center;
            size = b.size;
            return true;
        }

        return false;
    }

    // Colliderの8頂点をAtom基準のローカル空間へ変換
    private void EncapsulateCollider(
        Collider col,
        Transform rootTx,
        ref Bounds bounds,
        ref bool initialized)
    {
        Vector3 center;
        Vector3 size;

        if (!GetColliderLocalBounds(col, out center, out size))
            return;

        Vector3 half = size * 0.5f;

        for (int x = -1; x <= 1; x += 2)
        {
            for (int y = -1; y <= 1; y += 2)
            {
                for (int z = -1; z <= 1; z += 2)
                {
                    Vector3 localCorner = center + new Vector3(
                        half.x * x,
                        half.y * y,
                        half.z * z);

                    Vector3 worldCorner =
                        col.transform.TransformPoint(localCorner);

                    Vector3 rootLocalCorner =
                        rootTx.InverseTransformPoint(worldCorner);

                    if (!initialized)
                    {
                        bounds = new Bounds(
                            rootLocalCorner, Vector3.zero);
                        initialized = true;
                    }
                    else
                    {
                        bounds.Encapsulate(rootLocalCorner);
                    }
                }
            }
        }
    }

    // Log the physical colliders individually without changing UDP or bounds computation.
    private void LogColliderDiagnostics(Atom atom, Transform rootTx, Collider[] colliders)
    {
        if (!colliderDiagnosticsEnabled) return;

        int atomId = GetAtomId(atom);
        if (!diagnosedAtomIds.Add(atomId)) return;

        int active = 0;
        int ignored = 0;
        int supported = 0;
        StringBuilder sb = new StringBuilder();
        sb.AppendFormat("[EnvCollider] Atom={0} ID={1} Found={2}\n",
            atom.name, atomId, colliders.Length);

        for (int i = 0; i < colliders.Length; i++)
        {
            Collider col = colliders[i];
            if (col == null) { ignored++; continue; }
            if (!IsPhysicalCollider(col))
            {
                ignored++;
                continue;
            }

            if (!IsTransferCollider(col))
            {
                ignored++;
                sb.AppendFormat("  [{0}] MeshCollider(non-convex) path={1} SKIPPED (large mesh bounds)\n",
                    i, col.gameObject.name);
                continue;
            }
            active++;
            Vector3 center;
            Vector3 size;
            bool recognized = GetColliderLocalBounds(col, out center, out size);
            if (recognized) supported++;

            // Avoid GetType(): VaM sandbox prohibits reflection metadata access.
            string shape = "OtherCollider";
            if (col is BoxCollider) shape = "BoxCollider";
            else if (col is SphereCollider) shape = "SphereCollider";
            else if (col is CapsuleCollider) shape = "CapsuleCollider";
            else if (col is MeshCollider) shape = "MeshCollider";
            MeshCollider mesh = col as MeshCollider;
            if (mesh != null)
            {
                shape += mesh.convex ? "(convex)" : "(non-convex)";
                if (mesh.sharedMesh != null)
                {
                    // vertexCount is metadata; triangles may be inaccessible on some meshes.
                    shape += string.Format(" verts={0}", mesh.sharedMesh.vertexCount);
                    try
                    {
                        shape += string.Format(" tris={0}", mesh.sharedMesh.triangles.Length / 3);
                    }
                    catch (Exception)
                    {
                        shape += " tris=unavailable";
                    }
                }
            }

            if (recognized)
            {
                // AABB in Atom-root coordinates. Used for diagnosis only.
                Bounds localBox = new Bounds();
                bool hasBounds = false;
                EncapsulateCollider(col, rootTx, ref localBox, ref hasBounds);
                sb.AppendFormat("  [{0}] {1} path={2} localCenter={3} localSize={4} rootBoxCenter={5} rootBoxSize={6}\n",
                    i, shape, col.gameObject.name,
                    center.ToString("F3"), size.ToString("F3"),
                    localBox.center.ToString("F3"), localBox.size.ToString("F3"));
            }
            else
            {
                sb.AppendFormat("  [{0}] {1} path={2} UNSUPPORTED\n",
                    i, shape, col.gameObject.name);
            }
        }

        sb.AppendFormat("  Summary: active={0} supported={1} ignored={2}",
            active, supported, ignored);
        SuperController.LogMessage(sb.ToString());
    }

    // One shape record = 12 float32 values:
    // source collider index, type (1=box, 2=sphere, 3=capsule, 4=mesh-bounds-box),
    // offset XYZ, full size XYZ, local quaternion XYZW.
    // All coordinates and lengths are in the ORIGINAL Unity world axes, with the
    // relative offset expressed in the Atom root's unrotated frame.
    // The offset is relative to the legacy 9001 pose CENTER (combined bounds center),
    // NOT to the Atom's mainController pivot. This is critical for alignment.
    // Receiver: Newton basis + height scaling are intentionally NOT applied here.
    private bool SendColliderShapes(Atom atom, Transform rootTx, Collider[] colliders,
                                   Vector3 worldCenter)
    {
        List<float> data = new List<float>();
        data.Add(ShapeMagic);
        data.Add(ShapeVersion);
        data.Add((float)GetAtomId(atom));
        data.Add(0f); // number of records, filled below
        int count = 0;
        Quaternion invRoot = Quaternion.Inverse(rootTx.rotation);

        for (int i = 0; i < colliders.Length; i++)
        {
            Collider col = colliders[i];
            if (!IsTransferCollider(col)) continue;
            Vector3 center, size;
            if (!GetColliderLocalBounds(col, out center, out size)) continue;

            int kind = 0;
            if (col is BoxCollider) kind = 1;
            else if (col is SphereCollider) kind = 2;
            else if (col is CapsuleCollider) kind = 3;
            else if (col is MeshCollider) kind = 4;
            if (kind == 0) continue;

            // These vectors contain the scale of all parent transforms.
            // Widths remain fixed after first registration, as in the old XML system.
            Vector3 ex = col.transform.TransformVector(Vector3.right);
            Vector3 ey = col.transform.TransformVector(Vector3.up);
            Vector3 ez = col.transform.TransformVector(Vector3.forward);
            Vector3 fullSize = new Vector3(size.x * ex.magnitude,
                                           size.y * ey.magnitude,
                                           size.z * ez.magnitude);
            if (fullSize.x <= 0f || fullSize.y <= 0f || fullSize.z <= 0f) continue;

            Vector3 colWorldCenter = col.transform.TransformPoint(center);
            Vector3 relativeCenter = invRoot * (colWorldCenter - worldCenter);
            Quaternion relativeRotation = invRoot * col.transform.rotation;

            data.Add((float)i);
            data.Add((float)kind);
            data.Add(relativeCenter.x);
            data.Add(relativeCenter.y);
            data.Add(relativeCenter.z);
            data.Add(fullSize.x);
            data.Add(fullSize.y);
            data.Add(fullSize.z);
            data.Add(relativeRotation.x);
            data.Add(relativeRotation.y);
            data.Add(relativeRotation.z);
            data.Add(relativeRotation.w);
            count++;
        }

        if (count == 0) return false;
        data[3] = (float)count;
        float[] floats = data.ToArray();
        byte[] bytes = new byte[floats.Length * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        try
        {
            udpClient.Send(bytes, bytes.Length, ipAddress, shapePort);
            SuperController.LogMessage(string.Format(
                "[EnvShape] Atom={0} ID={1} colliders={2} bytes={3} -> UDP:{4}",
                atom.name, GetAtomId(atom), count, bytes.Length, shapePort));
            return true;
        }
        catch (Exception e)
        {
            SuperController.LogError("[EnvShape] UDP send failed: " + e.Message);
            return false;
        }
    }

    void Update()
    {
        timeSinceLastSend += Time.deltaTime;
        bool sendPose = timeSinceLastSend >= sendInterval;
        if (sendPose) timeSinceLastSend = 0f;

        List<int> validIds = new List<int>();
        List<Vector3> validCenters = new List<Vector3>();
        List<Vector3> validSizes = new List<Vector3>();
        List<Quaternion> validRotations = new List<Quaternion>();

        List<Atom> allAtoms = SuperController.singleton.GetAtoms();
        foreach (Atom atom in allAtoms)
        {
            if (atom == null) continue;
            // Exclude system/UI objects, but accept any eligible furniture Atom.
            if (!IsEligibleAtom(atom)) continue;

            Collider[] colliders = atom.GetComponentsInChildren<Collider>();
            if (colliders == null || colliders.Length == 0) continue;
            Transform rootTx = atom.mainController != null
                ? atom.mainController.transform : atom.transform;
            Bounds localBounds = new Bounds();
            bool initialized = false;
            foreach (Collider col in colliders)
            {
                if (!IsTransferCollider(col)) continue;
                EncapsulateCollider(col, rootTx, ref localBounds, ref initialized);
            }
            if (!initialized) continue;

            Vector3 worldCenter = rootTx.TransformPoint(localBounds.center);
            Vector3 scale = rootTx.lossyScale;
            Vector3 worldSize = new Vector3(
                localBounds.size.x * Mathf.Abs(scale.x),
                localBounds.size.y * Mathf.Abs(scale.y),
                localBounds.size.z * Mathf.Abs(scale.z));
            // Retain the original filter, shared by shape and pose packets.
            if (worldCenter.y <= 0f || worldSize.x > 8f || worldSize.z > 8f)
                continue;

            LogColliderDiagnostics(atom, rootTx, colliders);
            // Send geometry once per Atom UID, immediately when first detected.
            // This also detects new Atoms without waiting for the 0.5s pose tick.
            if (!shapeSentUids.Contains(atom.uid))
            {
                if (SendColliderShapes(atom, rootTx, colliders, worldCenter))
                    shapeSentUids.Add(atom.uid);
            }

            if (!sendPose) continue;
            validIds.Add(GetAtomId(atom));
            validCenters.Add(worldCenter);
            validSizes.Add(worldSize);
            validRotations.Add(rootTx.rotation);
        }

        if (!sendPose) return;
        int count = validIds.Count;
        float[] packetData = new float[1 + count * 11];
        packetData[0] = (float)count;
        for (int i = 0; i < count; i++)
        {
            int offset = 1 + i * 11;
            Vector3 center = validCenters[i];
            Vector3 size = validSizes[i];
            Quaternion rot = validRotations[i];
            packetData[offset + 0] = (float)validIds[i];
            packetData[offset + 1] = center.x;
            packetData[offset + 2] = center.y;
            packetData[offset + 3] = center.z;
            packetData[offset + 4] = size.x;
            packetData[offset + 5] = size.y;
            packetData[offset + 6] = size.z;
            packetData[offset + 7] = rot.x;
            packetData[offset + 8] = rot.y;
            packetData[offset + 9] = rot.z;
            packetData[offset + 10] = rot.w;
        }
        try
        {
            byte[] bytes = new byte[packetData.Length * 4];
            Buffer.BlockCopy(packetData, 0, bytes, 0, bytes.Length);
            udpClient.Send(bytes, bytes.Length, ipAddress, port);
        }
        catch (Exception e)
        {
            SuperController.LogError("[EnvSender] Pose UDP send failed: " + e.Message);
        }
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}
