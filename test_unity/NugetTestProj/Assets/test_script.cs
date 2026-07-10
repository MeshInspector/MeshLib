using UnityEngine;

public class test_script : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        MR.DecimateSettings dp = new MR.DecimateSettings();
        Debug.Log(dp);
        Debug.Log("MeshTestPassed");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
