本文档旨在指导 Android 开发者如何将由固件厂商或平台方提供的 `.pk8` 私钥文件和 `.pem` 证书文件，转换为可用于 APK 签名的标准 Java Keystore (`.keystore`) 文件。

## 准备工作

在开始之前，确保开发环境中已安装以下工具：

1. **OpenSSL**: 用于处理和转换密钥/证书格式。
   * macOS/Linux: 通常系统自带。
   * Windows: 可通过 Git for Windows (Git Bash) 或 [独立安装包](https://slproweb.com/products/Win32OpenSSL.html) 获取。
   * 验证命令: `openssl version`
2. **Java Development Kit (JDK)**: `keytool` 工具包含在 JDK 中。
   * Android Studio 自带 JDK，你也可以单独安装。
   * 验证命令: `keytool -help`

**假设你的文件名为:**

* 私钥文件: `platform.pk8`
* 证书文件: `platform.x509.pem`

---

## 核心问题：DER vs PEM 格式

直接使用 `openssl` 命令转换时，可能会遇到类似 `Could not find private key...` 的错误。这通常是因为 `.pk8` 文件是 **DER** 编码的二进制格式，而许多 OpenSSL 命令默认期望 **PEM** 编码的文本格式。

* **DER 格式**: 二进制格式，用文本编辑器打开会看到乱码。
* **PEM 格式**: Base64 编码的文本格式，内容由 `-----BEGIN...-----` 和 `-----END...-----` 包裹。

我们的第一步就是解决这个格式问题。

---

## 操作步骤

### 第一步：将 DER 格式的私钥转换为 PEM 格式

在终端中，进入文件所在目录，执行以下命令：

```bash
openssl pkcs8 -topk8 -inform DER -in platform.pk8 -outform PEM -nocrypt -out platform_converted.pk8
```

**参数说明:**

* `-inform DER`: **明确指定**输入文件 (`platform.pk8`)是 DER 格式。
* `-in platform.pk8`: 指定原始的私钥文件。
* `-outform PEM`: 指定输出文件为 PEM 格式。
* `-nocrypt`: 指定输出的私钥不进行加密，简化后续步骤。
* `-out platform_converted.pk8`: 指定转换后新生成的文件名。

执行成功后会得到一个新文件 `platform_converted.pk8`，其内容是以 `-----BEGIN PRIVATE KEY-----` 开头的文本。

### 第二步：将 PEM 私钥和证书合并为 PKCS12 文件 (.p12)

现在，使用上一步生成的新私钥文件和原始的证书文件来创建一个 `.p12` 文件。

```bash
openssl pkcs12 -export -in platform.x509.pem -inkey platform_converted.pk8 -out platform.p12 -name mykeyalias
```

**参数说明:**

* `-export`: 表示创建一个 PKCS12 文件。
* `-in platform.x509.pem`: 指定输入的证书文件。
* `-inkey platform_converted.pk8`: **注意！** 指定输入的私钥文件是上一步**转换后**的文件。
* `-out platform.p12`: 指定输出的 PKCS12 文件名。
* `-name mykeyalias`: 为密钥对设置一个**别名 (alias)**。这个别名非常重要，请记下它（例如 `platformkey`）。

执行此命令后，系统会提示设置一个**导出密码 (Export Password)**。**请务必记住此密码**，下一步会用到。

### 第三步：将 PKCS12 文件转换为 Java Keystore (.keystore)

这是最后一步，使用 Java 的 `keytool` 工具生成最终的 `.keystore` 文件。

```bash
keytool -importkeystore \
        -deststorepass <你的Keystore密码> \
        -destkeypass <你的密钥密码> \
        -destkeystore my-release-key.keystore \
        -srckeystore platform.p12 \
        -srcstoretype PKCS12 \
        -srcstorepass <第二步设置的p12密码> \
        -alias mykeyalias
```

**参数说明:**

* `-deststorepass`: 设置新生成的 `.keystore` 文件的密码。
* `-destkeypass`: 设置该密钥在 Keystore 中的密码 (通常与 `-deststorepass` 设为相同以方便管理)。
* `-destkeystore`: 指定最终生成的 Keystore 文件名。
* `-srckeystore platform.p12`: 指定源文件为上一步生成的 `.p12` 文件。
* `-srcstoretype PKCS12`: 指定源文件类型为 PKCS12。
* `-srcstorepass`: 输入你在第二步中为 `.p12` 文件设置的**导出密码**。
* `-alias mykeyalias`: **必须与**第二步中 `-name` 参数设置的别名**完全一致**。

执行成功后，`my-release-key.keystore` 文件就是你最终需要的安卓签名文件。

---

## 验证与使用

### 验证 Keystore

你可以使用以下命令验证 Keystore 是否创建成功，并查看其内容。

```bash
keytool -list -v -keystore my-release-key.keystore
```

系统会提示你输入 Keystore 密码，然后显示密钥条目的详细信息。

### 在 Android Studio 中使用

1. 将 `my-release-key.keystore` 文件复制到项目的 `app` 目录下。
2. 在 `app/build.gradle` 文件中配置 `signingConfigs`。

```groovy
android {
    // ...

    signingConfigs {
        release {
            storeFile file('my-release-key.keystore')
            storePassword '你的Keystore密码'
            keyAlias 'mykeyalias'
            keyPassword '你的密钥密码'
        }
    }

    buildTypes {
        release {
            signingConfig signingConfigs.release
            // ...
        }
    }
}
```

**安全注意**: 避免在 `build.gradle` 中硬编码密码。推荐将密码等信息存储在项目根目录的 `gradle.properties` 文件中，并从 `build.gradle` 中引用。

---

## 完整流程总结

`DER (.pk8)` + `.pem` → **`openssl pkcs8`** → `PEM (.pk8)` + `.pem` → **`openssl pkcs12`** → `.p12` → **`keytool`** → `.keystore`


