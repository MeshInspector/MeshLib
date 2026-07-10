using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;

public class mesh_test
{
    // A Test behaves as an ordinary method
    [Test]
    public void mesh_testSimplePasses()
    {
        // Use the Assert class to test conditions
        GameObject testObject = new GameObject("MeshlibObject");
        Component[] components = testObject.GetComponents<Component>();
        foreach (Component component in components)
        {
            Debug.Log(component.GetType().Name);
        }
        Debug.Log("componentNames");
    }

    [UnitySetUp]
    public IEnumerator SetUp()
    {
        if (SceneManager.GetActiveScene().name != "SampleScene")
        {
            // load the test scene before the tests
            yield return SceneManager.LoadSceneAsync("SampleScene");
        }
    }

        // A UnityTest behaves like a coroutine in Play Mode. In Edit Mode you can use
        // `yield return null;` to skip a frame.

    [UnityTest]
    public IEnumerator mesh_testWithEnumeratorPasses()
    {
        // Use the Assert class to test conditions.
        // Use yield to skip a frame.
        GameObject meshlibObject = GameObject.Find("MeshlibObject");
        Assert.IsNotNull(meshlibObject);
        // wait a frame so that Unity calls Start() on scene objects
        yield return null;
        Debug.Log("Custom test passed");
    }
}
